# SituLM
## Installation
1. Clone this repository and navigate to LLaVA-backbone folder
```bash
git clone https://github.com/PYL2077/SituLM.git
cd SituLM/LLaVA-backbone
```

2. Install Package
```Shell
conda create -n situlm python=3.10 -y
conda activate situlm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## LLaVA Weights
Click [here]([https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md](https://huggingface.co/liuhaotian/llava-v1.5-7b)) to download pre-trained LLaVA weights. 

## CLI Inference

Perform custom inference with SituLM. It also supports multiple GPUs, 4-bit and 8-bit quantized inference.
```Shell
python -m llava.serve.cli \
    --model-path /path/to/llava-v1.5-7b \
    --image-file /path/to/image.png
    --load-4bit
```
