# Who Is Speaking (WIS)

## Introduction
A popular approach for face recogintion tasks is training a deep model using Triplet Margin Loss in order to construct face embeddings. Based on the embeddings one shot learning can be performed using a Siamese Network to classify faces. An overview of the approach can be fund [here](https://machinelearningmastery.com/one-shot-learning-with-siamese-networks-contrastive-and-triplet-loss-for-face-recognition/).

This project applyies the concept to voice recordings in order to enable one shot learning speaker classification.

## Data
The model is trained on the small version of the [LibriSpeech ASR corpus](http://www.openslr.org/12/) containing 100 hours of of audiobook data from 250 different readers. 

In order to feed the audio data into the CNN it firstly has to be transformed into an image. For this a Mel Spectrogram in dB is used, which is a Spectrogram where the frequency axis is mapped to Mel scale in dB. Mel better approximates the way humans perceive sound. dB is used as frequencies not audible for humans can be dropped. The number of frequency bins is 128.

As recordings in the dataset are of different duration, the Spectrograms vary on the y-axis. Therefore they are decomposed to have y-axis length 128 (about 2.9s).

Example of dataset instance transformed to Mel Spectrum in dB:
<img src="https://github.com/dubachti/WIS/blob/da01290fbfd102b19edaead72558e289fec3a529/readme_img/mel_spectrogram.png" alt= “” width="50%" height="50%">

## Model
The model used is ResNet18 with an additional top layer to match the data instance shape and bottom layer to create the embeddings of dim 32. Training is performed using the Triplet Margin Loss.


## Train / Evaluate
Firtsly the data instances have to be transofrmed into Mel Spectrograms of shape (128,128). This can be achieved using the command
```bash
python3 src/preprocessing.py --path PATH/TO/DATA
```

Using the command
```bash
python3 src/train_model.py --path PATH/TO/TRANSFORMED_DATA --train True
```
it is possible to train the model. Using the flags **--lr**, **--epoch** and **--num_workers** further parameters can be adjusted. Add the end of training the weights will be stored in **weights/**.

Model performance can be evaluated using the command
```bash
python3 src/train_model.py --path PATH/TO/TRANSFORMED_DATA --train False
```
Note that the used criterion is the Euclidean distance in contrast to the Triplet Margine Loss used in the training and testing of the model.

## GUI
The command
```bash
python3 src/gui.py
```
opens a simple tkinter GUI which lets you save speakers using your microphone and then predicts the speaker upon receiving a new voice recording.

Screenshot of the GUI predicting a newly recorded voice:
<img src="https://github.com/dubachti/WIS/blob/da01290fbfd102b19edaead72558e289fec3a529/readme_img/gui.png" alt= “” width="50%" height="50%">