import re
import numpy as np
import sounddevice as sd
import torch
import threading
import pyaudio
import time
import queue
from pathlib import Path
from llama_cpp import Llama
from kokoro import KModel, KPipeline
import configparser
import sherpa_onnx

# 全局变量用于控制麦克风状态
mic_enabled = True
mic_lock = threading.Lock()  # 用于同步 mic_enabled 变量


# 检查文件是否存在
def assert_file_exists(filename: str):
    assert Path(filename).is_file(), f"{filename} does not exist!"


# 读取配置文件
config = configparser.ConfigParser()
config.read("config.ini")

# 获取路径配置
llm_model_path = config.get("Paths", "llm_model_path")
encoder_path = config.get("Paths", "encoder_path")
decoder_path = config.get("Paths", "decoder_path")
joiner_path = config.get("Paths", "joiner_path")
tts_en_voice = config.get("Paths", "tts_en_voice")
tokens_path = config.get("Paths", "tokens_path")


# 创建识别器
def create_recognizer():
    for file in [llm_model_path, encoder_path, decoder_path, joiner_path]:
        assert_file_exists(file)

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens_path,
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        max_active_paths=4,
        provider="cpu",
    )


# LLM 模型
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

llm_model: Llama = None
model_loaded = False

# Kokoro TTS 初始化
enpl = KPipeline(lang_code="a", model=None)
zhpl = KPipeline(lang_code="z", model=None)
koko_model = KModel()
speed = 1.0

# TTS 队列和锁
tts_queue = queue.Queue()
tts_lock = threading.Lock()


# 加载 LLM 模型
def load_llm_model():
    global llm_model, model_loaded

    if not model_loaded:
        with torch.no_grad():
            llm_model = Llama(
                model_path=llm_model_path,
            )
            model_loaded = True


# TTS推理函数
def infer(text, voice, speed, model):
    pack = zhpl.load_voice(voice)
    pack = pack.to(model.device) if model else pack
    ts = text.strip().split("\n")
    pt = r"[\u3100-\u9fff]+|[^\u3100-\u9fff]+"
    for t in ts:
        token_list = []
        ta = re.findall(pt, t)
        for t1 in ta:
            if is_chinese(t1[0]):
                token_list.append(zhpl.g2p(t1))
            else:
                _, tokens = enpl.g2p(t1)
                for gs, ps, m_tokens in enpl.en_tokenize(tokens):
                    if ps:
                        token_list.append(ps)
        ps = "".join(token_list)
        if len(ps) > 510:
            ps = ps[:510]
        output = KPipeline.infer(model, ps, pack, speed)
        audio_tensor = output.audio
        audio = audio_tensor.detach().cpu().numpy()
        if isinstance(audio, np.ndarray):
            yield audio


# 播放音频
def play_audio(audio):
    sd.play(audio, 24000)
    sd.wait()


# 判断是否为中文字符
def is_chinese(c):
    return (
        "\u4e00" <= c <= "\u9fff"
        or "\u3000" <= c <= "\u303f"
        or "\uff00" <= c <= "\uffef"
    )


# LLM推理
def llm_inference(message, history: list[dict[str, str]]):
    # 初始化对话消息
    messages = [
        {
            "role": "system",
            "content": "You are a friendly Assistant.",
        },
    ]

    # 根据历史记录添加对话内容
    if history:
        # 先添加用户输入
        messages.extend(history)
        messages.append({"role": "user", "content": message})
    else:
        messages.append({"role": "user", "content": message})
    history = messages
    response = llm_model.create_chat_completion(
        messages=messages,
        stream=True,
    )

    text = ""
    full_text = ""
    for chunk in response:
        try:
            delta = chunk["choices"][0]["delta"]
            if "content" not in delta:
                continue
            content = delta["content"]
            print(content, end="", flush=True)
            text += content
            full_text += content
            # 每次生成文本后，检查是否为完整的一句
            if content[-1] in "。！？！、":
                # print(f"\n触发TTS: {text}")
                # 将句子放入 TTS 队列
                tts_queue.put(text)

                text = ""  # 清空当前已生成的文本
        except AttributeError as e:
            print(f"处理响应时发生错误: {e}")
    history.append({"role": "assistant", "content": full_text})
    return history


# TTS 处理线程
def tts_worker():
    global mic_enabled  # 添加全局变量引用
    while True:
        text = tts_queue.get()
        if text is None:  # 退出机制
            break
        with tts_lock:
            # 播放前禁用麦克风
            mic_enabled = False
            play_streaming_tts(text)
            # 播放完成后重新启用麦克风
            mic_enabled = True


# 逐句播放 TTS（标点符号时触发）
def play_streaming_tts(text):
    # 使用正则表达式分句，同时保留标点符号
    sentences = re.split(r"([，。！？；])", text)  # 依据标点分割
    sentences = [
        sentence.strip() + mark if sentence.strip() else mark
        for sentence, mark in zip(sentences[::2], sentences[1::2])
    ]

    for sentence in sentences:
        # 每次处理一个句子
        g = infer(sentence.strip(), tts_en_voice, speed, koko_model)
        for a in g:
            if isinstance(a, np.ndarray):
                play_audio(a)  # 直接播放每个音频片段


# 音频监听线程
def listen_for_audio(history):
    global mic_enabled  # 引入全局变量引用
    pyaudio_instance = pyaudio.PyAudio()
    try:
        stream = pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096,
        )

        while True:
            with mic_lock:  # 在访问 mic_enabled 时加锁
                if not mic_enabled:
                    stream.stop_stream()
                    time.sleep(0.5)  # 避免频繁检查
                    continue

            recognizer = create_recognizer()
            sample_rate = 16000
            samples_per_read = int(0.1 * sample_rate)
            last_result = ""
            last_time = time.time()
            stream = recognizer.create_stream()

            with sd.InputStream(
                channels=1, dtype="float32", samplerate=sample_rate
            ) as s:
                while True:
                    samples, _ = s.read(samples_per_read)
                    samples = samples.reshape(-1)
                    stream.accept_waveform(sample_rate, samples)

                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)

                    result = recognizer.get_result(stream)
                    if result != last_result:
                        last_result = result
                        print("\r{}".format(result), end="", flush=True)
                        last_time = time.time()

                    if time.time() - last_time > 3 and last_result != "":
                        recognized_text = result
                        last_result = ""
                        result = ""
                        stream = recognizer.create_stream()
                        last_time = time.time()

                        history = llm_inference(recognized_text, history)
                        print(history)

    except OSError as e:
        print(f"错误: {e}")
        stream.close()
        time.sleep(0.2)
        mic_enabled = True
        stream = pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096,
        )
    except Exception as e:
        mic_enabled = True
        print(f"未知错误: {e}")
        time.sleep(0.2)


# 启动线程
def start():
    history = []
    load_llm_model()
    tts_thread = threading.Thread(target=tts_worker)
    tts_thread.start()
    listen_for_audio(history)
    tts_thread.join()


if __name__ == "__main__":
    start()
