import os
from ultralytics import YOLO

#os.chdir('D:\TA_KITTI\17 - 19 Mar\pyai\code_pyai18_p2\workshop\my_image_food\YOLODataset')
#print(os.getcwd())

model = YOLO('yolo11n.pt')
data = os.getcwd() + '/dataset.yaml'

#results = model.train(data=data, epochs=200, device='cuda', dropout=0.4, imgsz=1024)
results = model.train(data=data, epochs=200, device='cpu', dropout=0.4, imgsz=512)
