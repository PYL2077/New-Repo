# SituLM
## Installation
1. Clone this repository
```bash
git clone https://github.com/PYL2077/SituLM.git
```

2. Install Package
```Shell
cd SituLM/LLaVA-backbone
conda create -n situlm python=3.10 -y
conda activate situlm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
cd ../
cd SAM
pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## LLaVA Weights
Click [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md](https://huggingface.co/liuhaotian/llava-v1.5-7b) to download pre-trained LLaVA weights. 

## SWiG Dataset
Click [here](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip) to download the SWiG dataset

## CLI Inference

Perform custom inference with SituLM. It also supports multiple GPUs, 4-bit and 8-bit quantized inference.
```Shell
python -m llava.serve.cli \
    --model-path /path/to/llava-v1.5-7b \
    --image-file /path/to/image.png
    --load-4bit
```
