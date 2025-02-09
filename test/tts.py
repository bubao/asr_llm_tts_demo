import re
import numpy as np
import sounddevice as sd
from kokoro import KModel, KPipeline
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# 初始化
enpl = KPipeline(lang_code="a", model=None)
zhpl = KPipeline(lang_code="z", model=None)
model = KModel()
speed = 1.0


# 判断是否为中文字符
def is_chinese(c):
    # 中文字符范围 + 中文标点符号范围
    return (
        "\u4e00" <= c <= "\u9fff"
        or "\u3000" <= c <= "\u303f"
        or "\uff00" <= c <= "\uffef"
    )


# TTS推理函数
def infer(text, voice, speed, model):
    # enpl = KPipeline(lang_code="a", model=None)
    # zhpl = KPipeline(lang_code="z", model=None)
    pack = zhpl.load_voice(voice)
    pack = pack.to(model.device) if model else pack
    ts = text.strip().split("\n")
    pt = r"[\u3100-\u9fff]+|[^\u3100-\u9fff]+"
    for i, t in enumerate(ts):
        token_list = []
        ta = re.findall(pt, t)
        for t1 in ta:
            if is_chinese(t1[0]):
                token_list.append(zhpl.g2p(t1))
            else:
                _, tokens = enpl.g2p(t1)
                # for gs, ps in enpl.en_tokenize(tokens):
                #     if not ps:
                #         continue
                #     else:
                #         token_list.append(ps)
                for gs, ps, m_tokens in enpl.en_tokenize(tokens):
                    if ps:  # 只处理非空的 token
                        token_list.append(ps)
        ps = "".join(token_list)
        if len(ps) > 510:
            ps = ps[:510]
        output = KPipeline.infer(model, ps, pack, speed)
        # 提取音频张量并转换为 numpy 数组
        audio_tensor = output.audio
        audio = audio_tensor.detach().cpu().numpy()  # 如果是GPU上训练，先移到CPU
        # 确保返回的是numpy数组
        if isinstance(audio, np.ndarray):
            yield audio


# 播放音频
def play_audio(audio):
    sd.play(audio, 24000)  # 假设采样率是24000
    sd.wait()  # 等待音频播放完毕


# 主循环
def main():
    while True:
        # 获取用户输入文本
        text = input("请输入需要转换为语音的文本: ")

        if text.strip().lower() == "exit":  # 输入 'exit' 时退出
            print("退出程序")
            break

        audios = []
        g = infer(text, "af_sky,zf_xiaoyi", speed, model)

        # 获取并拼接生成的音频
        for a in g:
            if isinstance(a, np.ndarray):  # 确保是音频数组
                audios.append(a)

        # 调试输出

        if audios:
            try:
                # 拼接音频数组
                audio = np.concatenate(audios)  # 拼接音频数组
                # 播放音频
                play_audio(audio)
            except ValueError as e:
                print(f"Error in concatenating audio: {e}")
                print("可能的原因是音频数组形状不匹配。")


if __name__ == "__main__":
    main()
