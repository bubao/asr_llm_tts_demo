# ASR LLM TTS Demo

## 项目简介

该项目实现了一个基于语音识别（ASR）、大语言模型（LLM）推理和语音合成（TTS）的自动化对话系统。用户可以通过语音输入与系统进行交互，系统通过 ASR 进行语音识别，然后利用 LLM 进行推理并生成响应，最后使用 TTS 播放生成的响应。

## 技术栈

- **ASR （自动语音识别）**：使用了 `sherpa-onnx` 库进行语音识别。
- **LLM （大语言模型）**：使用 `DeepSeek-R1-Distill-Qwen-1.5B` 模型进行推理。
- **TTS （语音合成）**：使用 `Kokoro TTS` 实现语音合成。
- **音频处理**：使用 `sounddevice` 和 `pyaudio` 处理音频输入和输出。
- **并发处理**：通过 `threading` 库启动多线程，实时监听音频并进行处理。

## 环境要求

- Python 3.11+
- macOS 或 Linux 系统（Windows 系统也支持）
- 必要的依赖包和资源

## 安装与配置

1. **创建虚拟环境**

   在项目目录下创建虚拟环境：
   ```bash
   python3.11 -m venv venv
   ```

2. **激活虚拟环境**

   - 在 Windows 上激活虚拟环境：
     ```bash
     venv\Scripts\activate
     ```

   - 在 macOS/Linux 上激活虚拟环境：
     ```bash
     source venv/bin/activate
     ```

3. **安装依赖模块**

   安装项目所需的所有依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. **下载并准备资源**

   - 创建 `resources` 目录：
     ```bash
     mkdir resources
     ```
   
   - 下载并解压 ASR 模型：
     ```bash
     wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
     tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
     mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 resources
     rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
     ```

## 启动项目

运行以下命令以启动应用：

```bash
python main.py
```

## 项目功能

1. **语音识别（ASR）**：通过麦克风实时捕捉音频并转换为文本。
2. **大语言模型（LLM）推理**：将识别到的文本输入大语言模型进行处理并生成响应。
3. **语音合成（TTS）**：将大语言模型生成的文本转化为语音并播放。

## 配置说明

- **ASR 模型**：使用 `sherpa-onnx` 进行语音识别。确保所有模型文件存在于 `resources` 文件夹中。
- **LLM 模型**：使用 `DeepSeek-R1-Distill-Qwen-1.5B` 作为大语言模型进行推理，模型加载在 `load_llm_model` 函数中进行。
- **TTS 模型**：使用 `Kokoro TTS` 进行中文和英文的语音合成，支持不同的语音。

## 项目结构

```tree
.
├── main.py              # 项目主文件，启动应用
├── requirements.txt      # 项目依赖
├── resources/            # 存放 ASR 模型资源
│   └── sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
├── kokoro/               # Kokoro TTS 相关代码
├── README.md             # 项目的 README 文件
└── ...                   # 其他代码文件
```

## 注意事项

- 确保您的计算设备支持 MPS（Apple Silicon 芯片）或者 CUDA（NVIDIA GPU），以便高效地运行 LLM 模型。
- `pyaudio` 和 `sounddevice` 是音频处理的关键模块，确保您的设备可以正确识别音频设备。

## 贡献

欢迎对本项目提出建议或提交 Pull Requests。如果您遇到问题，欢迎提 issues。
