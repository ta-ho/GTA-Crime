import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, category: str = "Normal"):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim        
        self.test_mode = test_mode
        self.category = category

        if test_mode == False:
            if category == 'Normal': 
                self.df = self.df.loc[self.df['label'] == 'Normal']
                self.df = self.df.reset_index()
            elif category == 'Shooting':
                self.df = self.df.loc[self.df['label'] == 'Shooting']
                self.df = self.df.reset_index()
            elif category == 'Fighting':
                self.df = self.df.loc[self.df['label'] == 'Fighting']
                self.df = self.df.reset_index()
            else:
                self.df = self.df.loc[self.df['label'] != 'Normal']
                self.df = self.df.reset_index()
                     
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])      
        # train 시 
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)     
        # test 시
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)    

        clip_feature = torch.tensor(clip_feature, dtype=torch.float32)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length        
    
    
class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, category: str = "A"):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.category = category
        
        if test_mode == False:
            if category == 'A': 
                self.df = self.df.loc[self.df['label'] == 'A']
                self.df = self.df.reset_index()
            elif category == 'B2':
                self.df = self.df.loc[self.df['label'].str.contains('B2')]
                self.df = self.df.reset_index()
            elif category == 'B1':
                self.df = self.df.loc[self.df['label'].str.contains('B1')]
                self.df = self.df.reset_index()
            else:
                self.df = self.df.loc[self.df['label'] != 'A']
                self.df = self.df.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature, dtype=torch.float32)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length
    

class GTAUCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, category: str = "Normal"):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim        
        self.test_mode = test_mode
        self.category = category
        
        if test_mode == False:
            if category == 'Normal': 
                self.df = self.df.loc[self.df['label'] == 'Normal']
                self.df = self.df.reset_index()
            elif category == 'Shooting':
                self.df = self.df.loc[self.df['label'] == 'Shooting']
                self.df = self.df.reset_index()
            elif category == 'Fighting':
                self.df = self.df.loc[self.df['label'] == 'Fighting']
                self.df = self.df.reset_index()
            else:
                self.df = self.df.loc[self.df['label'] != 'Normal']
                self.df = self.df.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])      
        # train 시 
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)     
        # test 시
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)    

        clip_feature = torch.tensor(clip_feature, dtype=torch.float32)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length
    
    
class GTAXDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, category: str = "A"):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim        
        self.test_mode = test_mode
        self.category = category
        
        if test_mode == False:
            if category == 'A': 
                self.df = self.df.loc[self.df['label'] == 'A']
                self.df = self.df.reset_index()
            elif category == 'B2':
                self.df = self.df.loc[self.df['label'].str.contains('B2')]
                self.df = self.df.reset_index()
            elif category == 'Fighting':
                self.df = self.df.loc[self.df['label'].str.contains('B1')]
                self.df = self.df.reset_index()
            else:
                self.df = self.df.loc[self.df['label'] != 'A']
                self.df = self.df.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])     

        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim) 

        clip_feature = torch.tensor(clip_feature, dtype=torch.float32)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length