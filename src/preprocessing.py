import os
import librosa
import numpy as np
import pickle
import argparse

# cuts audio singal to a length of approx. 3s and transforms it to a mel spectogram
def transform_and_store(path: str, destination_dir: str) -> None:
    y, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    file_name = path.split('/')[-1][:-5]
    reader = file_name.split('-')[0]

    img = S_dB[:,:128]
    if img.shape != (128,128): return # snipped not long enough

    with open(f'{destination_dir}_transformed/{file_name}', 'wb') as f:
        pickle.dump((reader,img),f)


def preprocessing(path: str) -> None:
    for root, _, files in os.walk(path):
        for file in files:
            if not file.endswith('.flac') or file.startswith('.'): continue
    
            with open(os.path.join(root, file), "r") as auto:
                transform_and_store(auto.name, path)
            print('/', end='')
    print()

def main():
    parser = argparse.ArgumentParser(description='Preproccess data')
    parser.add_argument('--path', default='data', type=str, help='path to data')
    args = parser.parse_args()
    preprocessing(args.path)

if __name__ == '__main__': main()