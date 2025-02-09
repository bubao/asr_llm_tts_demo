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
import time
import sherpa_onnx
from pathlib import Path


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
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        max_active_paths=4,
        provider="cpu",
    )


# LLM 模型
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
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
def llm_inference(input_text, history):
    # 维持对话历史
    prompt = '<|begin_of_sentence|>System: You are a friendly Assistant. You initiate your response with "<think>\\n嗯" at the beginning of every output.\n'
    for h in history:
        prompt += f"<|User|>{h[0]}\n<|Assistant|>{h[1]}\n"
    prompt += f"<|User|>{input_text}\n<|Assistant|>"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    attention_mask = inputs.attention_mask if "attention_mask" in inputs else None
    input_ids = inputs.input_ids.to(DEVICE)
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)
    outputs = llm_model.generate(
        input_ids,
        max_new_tokens=200,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = response.split("<|Assistant|>")[-1].strip()

    assistant_response = assistant_response.replace("</think>", "").strip()

    return assistant_response


# 音频监听线程
def listen_for_audio(history):
    pyaudio_instance = pyaudio.PyAudio()
    stream = pyaudio_instance.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=4096,
    )
    print("正在监听音频...")

    while True:
        try:
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

                        llm_response = llm_inference(recognized_text, history)
                        print(f"\nLLM 响应: {llm_response}")
                        history.append((recognized_text, llm_response))  # 保存历史记录
                        play_tts(llm_response)

        except OSError as e:
            print(f"错误: {e}")
            stream.close()
            time.sleep(1)
            stream = pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096,
            )
        except Exception as e:
            print(f"未知错误: {e}")
            time.sleep(1)


# 逐句播放 TTS
def play_tts(text):
    sentences = text.split(".")
    for sentence in sentences:
        audios = []
        g = infer(sentence.strip(), "af_sky,zf_xiaoyi", speed, koko_model)
        for a in g:
            if isinstance(a, np.ndarray):
                audios.append(a)

        if audios:
            try:
                audio = np.concatenate(audios)
                play_audio(audio)
            except ValueError as e:
                print(f"拼接音频时出错: {e}")


# 启动监听线程
def start_listening():
    history = []  # 用于保存历史记录
    load_llm_model()  # 加载 LLM 模型
    listen_thread = threading.Thread(target=listen_for_audio, args=(history,))
    listen_thread.start()


if __name__ == "__main__":
    start_listening()
