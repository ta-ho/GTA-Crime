import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        if self.dataset == 'UCF':
            if test_mode:
                self.rgb_list_file = 'list/UCF3_Test.list'
            else:
                self.rgb_list_file = 'list/UCF3_Train.list'


        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'UCF':
                if self.is_normal:
                    self.list = self.list[72:]
                else:
                    self.list = self.list[:72]

    def __getitem__(self, index):

        label = self.get_label() 
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)
        features = features.transpose(1, 0, 2) 
        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2) 
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32) 
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)
            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame



class GTADataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        if self.dataset == 'UCF':
            if test_mode:
                exit()
            else:
                self.rgb_list_file = 'list/newGTA.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'UCF':
                if self.is_normal:
                    self.list = self.list[270:]
                else:
                    self.list = self.list[270:]


    def __getitem__(self, index):

        label = self.get_label() 
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)
        features = features.transpose(0, 1, 2)  
        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            features = features.transpose(0, 1, 2)
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32) 
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)
            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame