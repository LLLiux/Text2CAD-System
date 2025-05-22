# ⚙️ Installation
## 🌍 Environment

- 🐧 Linux
- 🐍 Python >=3.9

## 📦 Dependencies

```bash
$ conda env create --file environment.yml
```

# 🤖 Inference

下载 checkpoint [here](https://huggingface.co/datasets/SadilKhan/Text2CAD/blob/main/text2cad_v1.0/Text2CAD_1.0.pth)，并放在 `App/checkpoint` 。（或放在其它位置，需修改 `Cad_VLM/config/inference_user_input.yaml` 的 `checkpoint_path` ）

下载 bert-large-uncased [here](https://huggingface.co/google-bert/bert-large-uncased/blob/main/pytorch_model.bin)，并放在 `App/bert-large-uncased`。

# 💻 Run Demo

```bash
$ cd App
$ gradio app.py
```
