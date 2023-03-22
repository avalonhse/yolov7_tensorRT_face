
python export_face.py --weights ../../data/yolo_face/models/yolov7.pt --grid --simplify

python main.py

docker run -it --name yolo --rm --gpus=all nvcr.io/nvidia/tensorrt:22.06-py3

cd data/yolo/models
docker cp yolov7.onnx yolo:/workspace/

./tensorrt/bin/trtexec --onnx=yolov7.onnx --fp16 --saveEngine=yolov7-fp16-1x8x8.engine --timingCacheFile=timing.cache
