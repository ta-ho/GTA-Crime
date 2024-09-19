import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from tqdm.auto import tqdm

def images2video(image_list, video_path, fps):
    image_list.sort()
    
    if len(image_list) != 0:
        frame = cv2.imread(image_list[0])
        height, width, layers = frame.shape
        
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for image in tqdm(image_list):
            video.write(cv2.imread(image))
            
        video.release()


folder_path = "D:/datasets/GTA_V_anomaly/data"
save_path = "D:/datasets/GTA_V_anomaly/videos_cv"
fps = 30

location_list = os.listdir(folder_path)
print(location_list)

for location in tqdm(location_list):
    print(location)
    for view in ['view1', 'view2']:
        path = os.path.join(os.path.join(folder_path, location), view)
        image_list = glob.glob(os.path.join(path, '*.tiff'))
        video_path = os.path.join(save_path, location+'_'+view+'.mp4')
        images2video(image_list, video_path, fps)
        
        
        
        