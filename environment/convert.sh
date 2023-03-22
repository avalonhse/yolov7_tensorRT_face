
# Yolo
python export.py --weights ../../data/yolo/models/yolov7.pt --grid --end2end --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640

docker run -it --name yolo --rm --gpus=all nvcr.io/nvidia/tensorrt:22.06-py3

cd data/yolo/models
docker cp yolov7.onnx yolo:/workspace/

# Export with FP16 precision, min batch 1, opt batch 8 and max batch 8
./tensorrt/bin/trtexec --onnx=yolov7.onnx --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --fp16 --workspace=4096 --saveEngine=yolov7-fp16-1x8x8.engine --timingCacheFile=timing.cache

# Test engine
# ./tensorrt/bin/trtexec --loadEngine=yolov7-fp16-1x8x8.engine

# Copy engine -> host:
docker cp yolo:/workspace/yolov7-fp16-1x8x8.engine .
mkdir yolov7
mkdir yolov7/1

echo 'name: "yolov7"
platform: "tensorrt_plan"
max_batch_size: 8
dynamic_batching { }' >> ./yolov7/config.pbtxt
cp yolov7-fp16-1x8x8.engine ./yolov7/1/model.plan

cd ..
docker run --gpus all --rm --ipc=host --shm-size=1g --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd)/yolov7:/models/yolov7 nvcr.io/nvidia/tritonserver:22.06-py3 tritonserver --model-repository=/models --strict-model-config=false --log-verbose 1

#Test
# docker run -it --ipc=host --net=host nvcr.io/nvidia/tritonserver:22.06-py3-sdk /bin/bash
# ./install/bin/perf_analyzer -m yolov7 -u 127.0.0.1:8001 -i grpc --shared-memory system --concurrency-range 16

cd ../../../yolov7_tensorRT_face/yolo/test
python client.py image data/dog.jpg