#import sys
#sys.path.append("\home\lb-jiawenxu\YOLOv8")
import sys
sys.path.append('/home/lb-jiawenxu/YOLOv8')

from ultralytics import YOLO
 
model = YOLO(r'\home\lb-jiawenxu\YOLOv8\ultralytics\runs\detect\train2\weights\best.pt')
model.export(format='onnx', optimize=True)
