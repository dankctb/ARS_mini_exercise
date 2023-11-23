
import cv2
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import importlib
import utils_config
from utils_config import *
importlib.reload(utils_config)

model = torch.hub.load("ultralytics/yolov5", 'custom',"best_exp2.pt")  # or yolov5n - yolov5x6, custom

capture = cv2.VideoCapture('/home/dank/sethust/20231/arsvn/mini_prj/raw_data/test.avi')

prev_check = 0
state_check = 0
cnt = 0
table_data = [
    ["orion", 0],
    ["tipo", 0],
    ["y40", 0],
    ["nestea_hoaqua", 0],
    ["chanh", 0],
    ["nestea_atiso", 0],
    ["karo", 0],
    ["jack-jill", 0],
    ["g7", 0],
    ["dilmah", 0]
]
detection = [
    ["Prev", ""],
    ["Current", ""],
]

video_infer_start = time.time()
while True:
    flag, frame0 = capture.read()
    if not flag :
      break
    
    frame = frame0[:,300:,:]
    scale_ratio = 4
    frame = cv2.resize(frame, None, fx = 1/scale_ratio, fy = 1/scale_ratio, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    start_time = time.time()

    check = check_object_frame(frame)

    if check > 0 : # object detected
        print('='*10,'object detected','='*10)
       
        if (check - prev_check) > 0 :
            results = model(frame)
            df = results.pandas().xyxy[0]
            frame0 = cv2.rectangle(frame0, (int(df['xmin'][0])*4+300,int(df['ymin'][0])*4), (int(df['xmax'][0])*4+300,int(df['ymax'][0])*4),
                            color = (0,0,255), thickness = 6)
        
            table_data[df['class'][0]][1] += 1 
            detection[0][1] = detection[1][1]
            detection[1][1] = df['name'][0]
            picture = draw(table_data, detection)
            frame0[:picture.shape[0],:picture.shape[1],] = picture
            cv2.imwrite(f'detection/FRAME_{cnt}_{detection[1][1]}.png',frame0)

    inference_time = (time.time() - start_time)*1000
    print(f'Infer time frame {cnt}: {inference_time:.5f} miliseconds')
    
    prev_check = check
    cnt +=1

    interrupt_key = cv2.waitKey(1)
    if interrupt_key == ord('q'):
        break

inference_time = time.time() - video_infer_start
print('='*10 ,f'Total inferring time : {inference_time:.5f} seconds', '='*10)
capture.release()
cv2.destroyAllWindows()