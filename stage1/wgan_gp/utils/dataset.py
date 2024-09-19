import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

class GTADataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map, category: str = "Normal"):
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
        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']

        return clip_feature, clip_label, clip_length


class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, category: str = "Normal"):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
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
        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length