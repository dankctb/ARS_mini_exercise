
import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import importlib
import utils_config
from utils_config import *
importlib.reload(utils_config)

model = torch.hub.load("ultralytics/yolov5", 'custom',"best_exp2.pt", device = 0)  # or yolov5n - yolov5x6, custom

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

while True:
    _, frame0 = capture.read()
    if not _ :
      break
    frame = frame0[:,300:,:]
    scale_ratio = 4
    frame = cv2.resize(frame, None, fx = 1/scale_ratio, fy = 1/scale_ratio, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    start_time = time.time()

    check = check_object_frame(frame)

    if ((prev_check - check ) == 0) & (state_check)  :
        print(f'Start Bounding Box frame_{cnt}')
        state_check = 0
        results = model(frame)
        df = results.pandas().xyxy[0] # bounding box info
        #results.show()

        table_data[df['class'][0]][1] += 1 # draw 2 table
        detection[1][1] = df['name'][0]
        picture = draw(table_data, detection)
        frame0[:picture.shape[0],:picture.shape[1],] = picture

        frame0 = cv2.rectangle(frame0, (int(df['xmin'][0])*4+300,int(df['ymin'][0])*4), (int(df['xmax'][0])*4+300,int(df['ymax'][0])*4),
                              color = (0,0,255), thickness = 4)
        detection[0][1] = df['name'][0]
        cv2.imwrite(f'detection/FRAME_{cnt}_{detection[1][1]}.png',frame0)

    if (prev_check - check ) == -1 : # False -> True
        print(f'===========Object detected frame_{cnt}================')
        state_check = 1

    inference_time = (time.time() - start_time)*1000
    print(f'Infer time frame {cnt}: {inference_time:.5f} miliseconds')
    prev_check = check
    cnt +=1