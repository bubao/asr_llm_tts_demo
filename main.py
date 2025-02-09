import re
import numpy as np
import sounddevice as sd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import threading
import pyaudio
import time  # 引入时间模块用于延迟
import sherpa_onnx
from pathlib import Path

"""
# 在项目目录下创建虚拟环境
python3.11 -m venv venv

# 在 Windows 上激活虚拟环境
venv\Scripts\activate

# 在 macOS/Linux 上激活虚拟环境
source venv/bin/activate

安装依赖模块:
pip install -f requirements.txt

mkdir resources
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 resources
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

启动：
python main.py
"""


# 检查文件是否存在
def assert_file_exists(filename: str):
    assert Path(filename).is_file(), f"{filename} does not exist!"


# 创建识别器
def create_recognizer():
    tokens = "./resources/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt"
    encoder = "./resources/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx"
    decoder = "./resources/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx"
    joiner = "./resources/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx"

    for file in [tokens, encoder, decoder, joiner]:
        assert_file_exists(file)

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=1,
        sample_rate=16000,  # 使用16000采样率，避免重采样
        feature_dim=80,
        decoding_method="greedy_search",
        max_active_paths=4,
        provider="cpu",
    )


# LLM 模型
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# 动态选择设备
if torch.cuda.is_available():
    DEVICE = "cuda"  # 如果支持 CUDA，则选择 GPU
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # 如果支持 MPS（Apple Silicon），则选择 MPS
else:
    DEVICE = "cpu"  # 如果都不支持，则使用 CPU
tokenizer = None
llm_model = None
model_loaded = False

# Kokoro TTS 初始化
from kokoro import KModel, KPipeline

enpl = KPipeline(lang_code="a", model=None)
zhpl = KPipeline(lang_code="z", model=None)
koko_model = KModel()
speed = 1.0


# 加载 LLM 模型
def load_llm_model():
    global llm_model, tokenizer, model_loaded
    if not model_loaded:
        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, trust_remote_code=True
            )
            llm_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
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
def llm_inference(input_text):
    prompt = f"<|begin_of_sentence|>System: You are a friendly Assistant.<|User|>{input_text}<|Assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # 显式设置 attention_mask，避免 pad_token 与 eos_token 冲突
    attention_mask = inputs.attention_mask if "attention_mask" in inputs else None

    input_ids = inputs.input_ids.to(DEVICE)
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)
    input_ids = inputs.input_ids.to("mps")
    # input_ids = inputs.input_ids
    outputs = llm_model.generate(
        input_ids,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|Assistant|>")[-1].strip()


# ASR和TTS监听线程
def listen_for_audio():
    pyaudio_instance = pyaudio.PyAudio()
    stream = pyaudio_instance.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=4096,  # 增加缓冲区大小
    )
    print("正在监听音频...")

    while True:
        try:
            recognizer = create_recognizer()
            # 音频采样率和读取设置
            sample_rate = 16000  # 使用16000采样率
            samples_per_read = int(0.1 * sample_rate)  # 每次读取 0.1 秒音频
            last_result = ""
            last_time = time.time()  # 记录最后一次识别时间
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
                        last_time = time.time()  # 重置计时器

                    # 如果3秒没有识别新结果，输出当前已识别的部分并清空句子
                    if time.time() - last_time > 3 and last_result != "":
                        # print("\n3秒未识别，输出短句: ", result)
                        recognized_text = result
                        last_result = ""  # 清空句子
                        result = ""
                        stream = recognizer.create_stream()  # 重置流
                        last_time = time.time()  # 重置计时器
                        llm_response = llm_inference(recognized_text)
                        print(f"LLM 响应: {llm_response}")
                        play_tts(llm_response)
                    # chunk = stream.read(4096)  # 增加每次读取的缓冲大小
                    # audio_data += chunk
                    # if len(audio_data) > 32000:  # 每次积累32KB数据作为一个块来识别
                    #     recognized_text = asr_recognize(audio_data)
                    #     print(f"识别到的文本: {recognized_text}")

                    #     # 将识别结果发给 LLM 模型
                    #     llm_response = llm_inference(recognized_text)
                    #     print(f"LLM 响应: {llm_response}")

                    #     # 执行 TTS 逐句播放
                    #     play_tts(llm_response)
        except OSError as e:
            print(f"错误: {e}")
            stream.close()  # 如果发生溢出错误，确保关闭流
            time.sleep(1)  # 等待后重启流
            stream = pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096,  # 重新打开流
            )
        except Exception as e:
            print(f"未知错误: {e}")
            time.sleep(1)  # 如果发生其他错误，稍作等待再继续


# 逐句播放 TTS
def play_tts(text):
    sentences = text.split(".")  # 简单分割为句子
    for sentence in sentences:
        audios = []
        g = infer(sentence.strip(), "af_sky,zf_xiaoyi", speed, koko_model)

        for a in g:
            if isinstance(a, np.ndarray):  # 确保是音频数组
                audios.append(a)

        if audios:
            try:
                audio = np.concatenate(audios)  # 拼接音频数组
                play_audio(audio)  # 播放音频
            except ValueError as e:
                print(f"拼接音频时出错: {e}")
                print("可能的原因是音频数组形状不匹配。")


# 启动监听线程
def start_listening():
    load_llm_model()  # 加载 LLM 模型
    listen_thread = threading.Thread(target=listen_for_audio)
    listen_thread.start()


if __name__ == "__main__":
    start_listening()
