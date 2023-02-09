import os
import librosa
import numpy as np
import pickle

def transform_and_store(path: str):
    y, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    file_name = path.split('/')[-1][:-5]
    reader = file_name.split('-')[0]

    img = S_dB[:,:128]
    if img.shape != (128,128): return # snipped not long enough

    with open(f'small_data_transformed/{file_name}', 'wb') as f:
        pickle.dump((reader,img),f)


def preprocessing(path: str):
    for root, _, files in os.walk(path):
        for file in files:
            if not file.endswith('.flac') or file.startswith('.'): continue
    
            with open(os.path.join(root, file), "r") as auto:
                transform_and_store(auto.name)
            print('/', end='')
    print()



preprocessing('small_data')