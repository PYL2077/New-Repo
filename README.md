# SituLM
## Installation
1. Clone this repository and navigate to LLaVA-backbone folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA-backbone
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .
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
