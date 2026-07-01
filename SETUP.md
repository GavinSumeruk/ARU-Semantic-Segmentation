# Environment Setup — NVIDIA Jetson Orin (R36.4.7 / JetPack 6.1)

## System info
- L4T: R36 (JetPack 6.1)
- Python: 3.10
- CUDA: 12.x (driver)
- TensorRT: 10.3.0 (system install at /usr/lib/python3.10/dist-packages/tensorrt)

## Step 1 — Clone the repo and create venv
```bash
git clone https://github.com/GavinSumeruk/ARU-Semantic-Segmentation
cd ARU-Semantic-Segmentation
python3 -m venv venv
source venv/bin/activate
```

## Step 2 — Install libcusparseLt (required for PyTorch 2.5+)
```bash
cd ~
curl -OL https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz
tar xf libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz
sudo cp -a libcusparse_lt-linux-aarch64-0.7.1.0-archive/include/* /usr/local/cuda/include/
sudo cp -a libcusparse_lt-linux-aarch64-0.7.1.0-archive/lib/* /usr/local/cuda/lib64/
sudo ldconfig
```

## Step 3 — Install PyTorch (JetPack 6.1 wheel)
```bash
cd ~/ARU-Semantic-Segmentation
pip install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip install "numpy<2"
```

## Step 4 — Make TensorRT visible inside the venv
```bash
echo "/usr/lib/python3.10/dist-packages" >> ~/ARU-Semantic-Segmentation/venv/lib/python3.10/site-packages/tensorrt.pth
```

## Step 5 — Build torchvision from source
```bash
cd ~
sudo apt-get install -y libjpeg-dev libpng-dev
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.20.0
source ~/ARU-Semantic-Segmentation/venv/bin/activate
nohup python3 setup.py install > ~/vision_build.txt 2>&1 &
tail -f ~/vision_build.txt
# wait ~25 minutes for build to complete
```

## Step 6 — Install remaining packages
```bash
cd ~/ARU-Semantic-Segmentation
pip install ultralytics timm onnx onnxslim
```

## Step 7 — Build YOLO TensorRT engine
```bash
yolo export model=yolo11n.pt format=engine half=True
# produces yolo11n.engine — device-specific, do not commit to Git
```

## Step 8 — Install NanoSAM (optional — engines pending)
```bash
cd ~
git clone https://github.com/NVIDIA-AI-IOT/nanosam
cd nanosam
pip install -e .
# ONNX files not yet available via public URL
# resnet18_image_encoder.onnx needs distilled checkpoint from NVIDIA
# mobile_sam_mask_decoder.onnx can be exported from MobileSAM checkpoint
```

## Verify everything works
```bash
cd ~/ARU-Semantic-Segmentation
source venv/bin/activate
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.5.0a0+... True
python3 main.py
# expect: detections + masks saved to output.jpg
```

## Notes
- Engine files (.engine) are device-specific — rebuild on each new Orin
- Model weights (.pt) are not in Git — downloaded automatically by Ultralytics on first run
- __pycache__ and venv are gitignored
- MobileSAM is used as SAM stand-in until NanoSAM engines are available
