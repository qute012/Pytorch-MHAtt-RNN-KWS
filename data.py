import librosa
import os
import numpy as np
import torch
import torch.utils.data as data

path = 'your path'

CLASSES = []
for _,dir,_ in os.walk(path):
    CLASSES = dir
    break
print(CLASSES)

class SolarDataset(data.Dataset):

    def __init__(self, mode='train', root=path):
        super(SolarDataset, self).__init__()
        self.root = os.path.join(root,mode)
        self.data = list()
        self.prep_dataset()
            
    def prep_dataset(self):
        for root, dir, files in os.walk(self.root):
            for file in files:
                f_path, cmd = os.path.join(root, file), root.split('/')[-1]
                self.data.append((f_path, cmd))

    def __getitem__(self, idx):
        f_path, cmd = self.data[idx]
        x = self.transform(f_path)
        y = CLASSES.index(cmd)
        return x, y

    def __len__(self):
        return len(self.data)
    
    def transform(self, path, sr=16000):
        sig, sr = librosa.load(path, sr)
        spec = librosa.feature.mfcc(sig, sr=sr,  n_mfcc=40)
        x = np.array(spec, np.float32, copy=False)
        x = torch.from_numpy(x)
        return x
    
def _collate_fn(batch):
    inputs = [s[0] for s in batch]
    targets = [s[1] for s in batch]
    
    B = len(batch)
    F,T = inputs[0].shape
    
    max_len = 0
    for input in inputs:
        max_len = max(max_len, len(input[0]))
    temp = torch.zeros(B,F,max_len)
    for x in range(B):
        temp[x,:,:inputs[x].size(1)] = inputs[x]
    inputs = temp.unsqueeze(1)
    targets = torch.LongTensor(targets)
    return inputs, targets
