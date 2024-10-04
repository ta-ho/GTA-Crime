import cv2
import numpy as np

import torch
#from clip import clip
from PIL import Image

import os
import glob
from tqdm.auto import tqdm

def video_crop(video_frame, type):
    # video_frame: (1799, 240, 320, 3)
    l = video_frame.shape[0]    # 1799
    new_frame = []
    for i in range(l):
        img = cv2.resize(video_frame[i], dsize=(340, 256))  #(240, 320, 3) -> (256, 340, 3)
        #new_frame.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        new_frame.append(img)       # 이미 video2npy에서 BGR2RGB 해주었음

    #1 
    img = np.array(new_frame)   # (1799, 256, 340, 3)
    if type == 0:
        img = img[:, 16:240, 58:282, :]
    #2 
    elif type == 1:
        img = img[:, :224, :224, :]
    #3
    elif type == 2:
        img = img[:, :224, -224:, :]
    #4
    elif type == 3:
        img = img[:, -224:, :224, :]
    #5
    elif type == 4:
        img = img[:, -224:, -224:, :]
    #6
    elif type == 5:
        img = img[:, 16:240, 58:282, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #7
    elif type == 6:
        img = img[:, :224, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #8
    elif type == 7:
        img = img[:, :224, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #9
    elif type == 8:
        img = img[:, -224:, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #10
    elif type == 9:
        img = img[:, -224:, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    
    return img      # (1799, 224, 224, 3)

def image_crop(image, type):
    img = cv2.resize(image, dsize=(340, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #1
    if type == 0:
        img = img[16:240, 58:282, :]
    #2
    elif type == 1:
        img = img[:224, :224, :]
    #3
    elif type == 2:
        img = img[:224, -224:, :]
    #4
    elif type == 3:
        img = img[-224:, :224, :]
    #5
    elif type == 4:
        img = img[-224:, -224:, :]
    #6
    elif type == 5:
        img = img[16:240, 58:282, :]
        img = cv2.flip(img, 1)
    #7
    elif type == 6:
        img = img[:224, :224, :]
        img = cv2.flip(img, 1)
    #8
    elif type == 7:
        img = img[:224, -224:, :]
        img = cv2.flip(img, 1)
    #9
    elif type == 8:
        img = img[-224:, :224, :]
        img = cv2.flip(img, 1)
    #10
    elif type == 9:
        img = img[-224:, -224:, :]
        img = cv2.flip(img, 1)
    
    return img

def video2npy(video_path, frames, num):
    cap = cv2.VideoCapture(video_path)
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    temp_npy = []
    video_npy = []
    
    i = 0
    
    while(1):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frames == num:
            if len(temp_npy) != 0:
               video_npy.append(temp_npy[-1])     # (240, 320, 3)
            temp_npy.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #video_npy.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        i += 1
        
    return np.stack(video_npy, axis=0, dtype=np.uint8)

"""
if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device)
    
    error_list = []
    
    # 영상 저장 경로
    root_folder_path = "D:/newGTAVideo"
    # 피처 저장 경로
    save_path = "D:/newGTA"
    os.makedirs(save_path, exist_ok=True)
    
    category_folder = os.listdir(root_folder_path)
    print(category_folder)
    
    for category in tqdm(category_folder):
        print()
        print(category)
        os.makedirs(os.path.join(save_path, category), exist_ok=True)
        print()
        video_list = glob.glob(os.path.join(root_folder_path, os.path.join(category, "*.mp4")))
        
        # 각 카테고리 폴더의 영상들
        for video in tqdm(video_list):
            video_npy = video2npy(video, 16, 0) # (1799, 240, 320, 3)
            print()
            print("video: ", video_npy.shape)
            print()
            
            # 제대로 뽑혔는지 확인을 위해 원래 피처 불러오기
            #org_feature = np.load(os.path.join(compare_folder_path, os.path.join(category, video.split('\\')[-1].split('.')[0] + '__' + str(0) + '.npy')))
            #print("orginal feature: ", org_feature.shape)
            
            #if(org_feature.shape[0] != video_npy.shape[0]):
            #    error_list.append(video.split('\\')[-1].split('.')[0])
            #else:
            # 크기가 같은 경우만 추출 다를 경우 추후에 추가
            # 영상들의 10 crop
            for i in range(10):
                corp_video_npy = video_crop(video_npy, i)                                   # (1799, 224, 224, 3)
                print("corp_video: ", corp_video_npy.shape)
                    
                # 16번째 프레임마다 추출
                    
                # 원래 피처와 추출 피처의 크기가 다를 경우 list에 저장
                    
                        
                video_features = torch.zeros(0).to(device)
                with torch.no_grad():
                    for j in range(0, corp_video_npy.shape[0]):
                        img = Image.fromarray(corp_video_npy[j])
                        img = preprocess(img).unsqueeze(0).to(device)
                        feature = model.encode_image(img)
                        video_features = torch.cat([video_features, feature], dim=0)
        
                video_features = video_features.detach().cpu().numpy()
                print("video feature: ", video_features.shape)
                    
                np.save(os.path.join(os.path.join(save_path, category), video.split('\\')[-1].split('.')[0] + '__' + str(i) + '.npy'), video_features)
                
    # 원래 피처와 추출 피처가 다른 경우 출력
    #print("original과 다른경우: ", len(error_list))
    #for err in error_list:
    #    print(err)
"""