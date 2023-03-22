import gdown

url = "https://drive.google.com/u/1/uc?id=1_ZjnNF_JKHVlq41EgEqMoGE2TtQ3SYmZ&export=download"
output = "../../data/yolo_face/models/yolov7.pt"
gdown.download(url, output)