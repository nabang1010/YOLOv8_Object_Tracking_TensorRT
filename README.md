# YOLOv8 DeepSORT TensorRT

Using OpenCV to capture video from camera or video file, then use YOLOv8 TensorRT to detect objects and DeepSORT TensorRT or BYTETrack to track objects. 

Support for both NVIDIA dGPU and Jetson devices.

## Performance

### Both OpenCV YOLOv8 and DeepSORT TensorRT
Using OpenCV to capture video from camera or video file, then use YOLOv8 TensorRT to detect objects and DeepSORT TensorRT to track objects.

| Model | Device | FPS |
| --- | --- | --- |
| OpenCV + YOLOv8n + DeepSORT | NVIDIA dGPU GTX 1660Ti 6Gb| ~ |
| OpenCV + YOLOv8n + DeepSORT | NVIDIA Jetson Xavier NX 8Gb | ~ |
| OpenCV + YOLOv8n + DeepSORT | NVIDIA Jetson Orin Nano 8Gb | ~44 |

### YOLOv8 TensorRT model

Test speed of YOLOv8 TensorRT model using `trtexec` from TensorRT

`/usr/src/tensorrt/bin/trtexec` on NVIDIA Jetson

> batch size = 1

| Model | Device | Throughput (qps) | Latency(ms) |
| --- | --- | --- | --- |
| `yolov8n.engine` | NVIDIA dGPU GTX 1660Ti 6Gb| ~419.742 | ~2.91736 |
| `yolov8n.engine` | NVIDIA Jetson Xavier NX 8Gb | ~ | ~ |
| `yolov8n.engine` | NVIDIA Jetson Orin Nano 8Gb | ~137.469 | ~137.469 |

### DeepSORT TensorRT model

Test speed of DeepSORT TensorRT model using `trtexec` from TensorRT

`/usr/src/tensorrt/bin/trtexec` on NVIDIA Jetson 

> batch size = 1

| Model | Device | Throughput (qps) | Latency(ms) |
| --- | --- | --- | --- |
| `deepsort.engine` | NVIDIA dGPU GTX 1660Ti 6Gb| ~614.738 | ~1.52197 | 
| `deepsort.engine` | NVIDIA Jetson Xavier NX 8Gb | ~ | ~ |
| `deepsort.engine` | NVIDIA Jetson Orin Nano 8Gb | ~546.135 | ~1.82227 |

## For NVIDIA dGPU

### Environment

- NVIDIA CUDA: 11.4
- NVIDIA TensorRT: 8.5.2


#### Clone repository

Clone repository and submodules

```bash
git clone --recurse-submodules https://github.com/nabang1010/YOLOv8_DeepSORT_TensorRT.git
```

#### Prepare enviroment

Create new enviroment

```bash
conda create -n yolov8_ds python=3.8
```

Activate enviroment

```bash
conda activate yolov8_ds
```

### Prepare models

Go to **`refs/YOLOv8-TensorRT`** and install requirements for exporting models

```bash
cd refs/YOLOv8-TensorRT
pip3 install -r requirements.txt
pip3 install tensorrt easydict pycuda lap cython_bbox
```
Install `python3-libnvinfer`

```bash
sudo apt-get install python3-libnvinfer
```

Download YOLOv8 weights from [ultralytics](https://github.com/ultralytics/ultralytics) here: [yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) and save in folder **`models/to_export`**

**Export YOLOv8 ONNX model**

In **`refs/YOLOv8-TensorRT`** run the following command to export YOLOv8 ONNX model

```bash
python3 export-det.py \
--weights ../../models/to_export/yolov8n.pt \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0
```

The output `.onnx` model will be saved in **`models/to_export`** folder, move the model to **`models/onnx`** folder 
```bash
mv ../../models/to_export/yolov8n.onnx ../../models/onnx/yolov8n.onnx
```
**Export YOLOv8 TensorRT model**

In **`refs/YOLOv8-TensorRT`** run the following command to export YOLOv8 TensorRT model

```bash
python3 build.py \
--weights ../../models/onnx/yolov8n.onnx \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--fp16  \
--device cuda:0
```
The output `.engine` model will be saved in **`models/onnx`** folder, move the model to **`models/trt`** folder 

```bash
mv ../../models/onnx/yolov8n.engine ../../models/engine/yolov8n.engine
```

**Build OpenCV**

```bash
bash build_opencv.sh
```

**Export DeepSORT TensorRT model *(if use BYTETrack, ignore this step)***


Install `libeigen3-dev`
```bash
apt-get install libeigen3-dev
```
Go to **`refs/deepsort_tensorrt`** and run the following command to build `onnx2engine`

```bash
cd refs/deepsort_tensorrt
mkdir build
cd build
cmake ..
make -j$(nproc)

```

> If catch error `fatal error: Eigen/Core: No such file or directory`, replace `#include <Eigen/*>` with `#include <eigen3/Eigen/*>` in all files of this repo (`datatype.h`, `kalmanfilter.cpp`) and rebuild again.

> If catch error `error: looser exception specification on overriding virtual function 'virtual void Logger::log(nvinfer1::ILogger::Severity`  add `noexcept` before `override` in `logger.h` line 239 and rebuild again.

Run following command to export DeepSORT TensorRT model

```bash
./build/onnx2engine ../../models/onnx/deepsort.onnx ../../models/engine/deepsort.engine
```
### Run script

**Go to `src` folder**

```bash
cd src
```

**Run YOLOv8 + DeepSORT**

```bash
python3 yolov8_deepsort_trt.py --show

```
**Run YOLOv8 + DeepSORT**

```bash
python3 yolov8_bytetrack_trt.py --show

```

## For NVIDIA Jetson Device

***Coming soon***


---

# References

- [ultralytics](https://github.com/ultralytics/ultralytics) 
- [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT)
- [deepsort_tensorrt](https://github.com/GesilaA/deepsort_tensorrt)
- [yolov5_deepsort_tensorrt](https://github.com/cong/yolov5_deepsort_tensorrt)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)



