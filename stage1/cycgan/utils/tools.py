import torch
import numpy as np
import os
import random

def get_batch_label(texts, prompt_text, label_map: dict):
    # texts: (128, 1) :string
    # prompt_text: ['normal', 'abuse', 'arrest', 'arson', ...]
    # label_map: {'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', ... }
    label_vectors = torch.zeros(0)
    if len(label_map) != 7:     # len(label_map)=14 (class가 14개니까...)
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            # (128, 1) string -> (128, 14) int : one-hot encoding
            for text in texts:      # 128개의 class를 (앞 64개는 Nomral, 뒤 64개는 13개 중 한 개)
                label_vector = torch.zeros(len(prompt_text))        # 
                if text in label_map:
                    label_text = label_map[text]        # 'Normal', 'Abuse', 'Arrest', ... 중 1개
                    label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    else:
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    label_text = label_map[label]
                    label_vector[prompt_text.index(label_text)] = 1
            
            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors    # (128, 14) label one-hot encoding

def get_prompt_text(label_map: dict):
    prompt_text = []
    for v in label_map.values():
        prompt_text.append(v)

    return prompt_text

# lengths: (t//256+1, ) 각 256 temporal에서 어디까지 인지 표시, maxlen=256
def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]           # temporal batch 크기
    mask = torch.empty(batch_size, maxlen)  # (temporal batch, maxlen) 크기의 mask 생성
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:             # maxlen 이전에 t가 끝날경우
            mask[i, lengths[i]:maxlen] = 1  # 포함되지 않는 부분=1
    
    return mask.bool()                      # 포함되는 부분: false, 포함되지 않는 부분: true

def random_extract(feat, t_max):
   r = np.random.randint(feat.shape[0] - t_max)
   return feat[r : r+t_max, :]

def uniform_extract(feat, t_max, avg: bool = True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), t_max+1, dtype=np.int32)
    if avg == True:
        for i in range(t_max):
            if r[i]!=r[i+1]:
                new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
            else:
                new_feat[i,:] = feat[r[i],:]
    else:
        r = np.linspace(0, feat.shape[0]-1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]
            
    return new_feat

def pad(feat, min_len):
    # feat: (t, 512), min_len=clip_dim, args.visaul-length=256 (t < 256)
    clip_length = feat.shape[0]     # t
    if clip_length <= min_len:
       return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)        # (256, 512) => t 뒤에 256-t 만큼 0으로 padding
    else:
       return feat      # (256, 512)

# train 시 사용
# (t, 512) feature를 (256, 512) feature와 t와 256 중 작은 값을 같이 return
def process_feat(feat, length, is_random=False):

        if is_random:
            return random_extract(feat, length), length
        # default
        else:
            return uniform_extract(feat, length), length    # (256, 512), 256, RTFM과 BN-WVAD에서 봤듯이 평균내서 길이를 256로 맞춤


# test 시 사용
def process_split(feat, length):
    # feat: (t, 512), length: clip_dim, args.visual-length = 256
    clip_length = feat.shape[0]     # t
    # t < 256, clip feature의 차원 t가 visual-length보다 작을 경우
    if clip_length < length:
        return pad(feat, length), clip_length       # (256, 512), t
    # t > 256, clip feature의 차원 t가 visual-length보다 클 경우
    else:
        split_num = int(clip_length / length) + 1       # t를 256으로 나눈다음 +1, i.e t=550이라고 가정->split_num=3
        for i in range(split_num):
            if i == 0:
                # feat[0:256, :].reshape(1, 256, 512)
                split_feat = feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])
            # i == 1
            elif i < split_num - 1:
                # feat[256:512, :].reshape(1, 256, 512) 한 다음 이전 split_feat이랑 concat
                split_feat = np.concatenate([split_feat, feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])], axis=0)
            # 마지막 남는 부분
            else:
                # 256보다 부족한 부분은 0으로 padding을 주어 (256, 512) 크기로 맞춘다.
                split_feat = np.concatenate([split_feat, pad(feat[i*length:i*length+length, :], length).reshape(1, length, feat.shape[1])], axis=0)

        return split_feat, clip_length      # (t//256 + 1, 256, 512), t
    
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.enabled = True
    
    
def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("Total iters: {}\n".format(test_info["total iters"]))