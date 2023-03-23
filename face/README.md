
In case of not exporting, skip A and use the model with folder data
```
cp -R data/yolov7 ../../data/yolo_face/models
```

and go to step B.

# A. Export

## A1. Convert pt to onnx
```
python export_face.py --weights ../../data/yolo_face/models/yolov7.pt --grid --simplify
```
## A2. Convert onnx to triton engine

### A.2.1 Run docker TensorRT
```
docker run -it --name yolo --rm --gpus=all nvcr.io/nvidia/tensorrt:22.06-py3
```
#### Test yolo.pt (optional)
```
python main.py
```
### A.2.2 Copy model to Docker
cd data/yolo/models
docker cp yolov7.onnx yolo:/workspace/ 

### A.2.3 Convert
```
./tensorrt/bin/trtexec --onnx=yolov7.onnx --fp16 --saveEngine=yolov7-fp16-1x8x8.engine --timingCacheFile=timing.cache
```
### A.2.4 Copy result from Docker to outside
```
docker cp yolo:/workspace/yolov7-fp16-1x8x8.engine .
```
### A.2.5 Create Model folder and config file
```
mkdir yolov7
mkdir yolov7/1

echo 'name: "yolov7"
platform: "tensorrt_plan"
max_batch_size: 8
dynamic_batching { }' >> ./yolov7/config.pbtxt
cp yolov7-fp16-1x8x8.engine ./yolov7/1/model.plan
```

# B. Run Triton Server
```
tmux new -s yolo

docker run --gpus all --name yolo --rm --ipc=host --shm-size=1g --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd)/yolov7:/models/yolov7 nvcr.io/nvidia/tritonserver:22.06-py3 tritonserver --model-repository=/models --strict-model-config=false --log-verbose 1
```
# C. Test
```
python client.py image data/selfie.jpg
```

