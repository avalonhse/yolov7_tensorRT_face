
cd yolov7-face
python models/export.py --weights ../data/yolo_face/models/yolov7-face.pt --grid 

python models/export.py --weights ../data/yolo_face/models/yolov7-tiny.pt --grid 

python export.py --weights ../../data/yolo_face/models/yolov7.pt --grid 
