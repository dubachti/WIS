from torch.utils.data.dataset import Dataset
import pickle
import torch
import random

class Data(Dataset):
    def __init__(self,
                 file_names: list,
                 transform = None) -> None:
        self.file_names = file_names
        self.transform = transform
        self.files_by_speaker = {}
        for file_name in file_names:
            speaker = file_name.split('/')[-1].split('-')[0]
            self.files_by_speaker.setdefault(speaker, []).append(file_name)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        file = self.file_names[index]

        # anchor
        with open(file, 'rb') as f:
            speaker, anchor = pickle.load(f)

        # positive
        pos_file = random.choice(self.files_by_speaker[speaker])
        with open(pos_file, 'rb') as f:
            _, positive = pickle.load(f)

        # negative
        other_speakers = list(self.files_by_speaker.keys())
        other_speakers.remove(speaker)
        neg_speaker = random.choice(other_speakers)
        neg_file = random.choice(self.files_by_speaker[neg_speaker])
        with open(neg_file, 'rb') as f:
            _, negative = pickle.load(f)
        

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        else:
            anchor = torch.tensor(anchor)
            positive = torch.tensor(positive)
            negative = torch.tensor(negative)

        return anchor, positive, negative
 
    def __len__(self) -> int:
        return len(self.file_names)
