import torch
from torch.nn.functional import pairwise_distance
import numpy as np
import librosa
from net import Net


class Embedding():
    def __init__(self) -> None:
        self.model = Net()
        self.model.load_state_dict(torch.load('weights/model_weights', map_location=torch.device('cpu')))
        self.model.eval()
        self.embeddings = {}

    def transform_to_mel(self, y: np.array, sr: int) -> np.array:
        y = y.squeeze()
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        middle = len(S_dB[0])//2
        img = S_dB[:,middle-64:middle+64]
        assert img.shape==(128,128), 'mel spectrogram shape does not match the model'
        return img

    def add_embedding(self, name: str, y: np.array, sr: int) -> None:
        img = self.transform_to_mel(y, sr)
        if not name:
            name = f'person_{self.num_embeddings()}'
        img = torch.tensor(img).view(1,1,128,128)
        self.embeddings[name] = self.model(img)

    def find_speaker(self, y: np.array, sr: int):
        img = self.transform_to_mel(y, sr)
        img = torch.tensor(img).view(1,1,128,128)
        searched = self.model(img)
        distances = [(name,pairwise_distance(searched, embedding)) for name, embedding in self.embeddings.items()] #change back  to gen
        name, _ = min(distances, key= lambda x: x[1])
        return name, distances

    def num_embeddings(self) -> int:
        return len(self.embeddings)


    