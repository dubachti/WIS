import os
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import numpy as np, cv2
import pickle



path = 'small_data_transformed'
# get file names for all data instances
file_names = []
for root, _, files in os.walk(path):
    for file in files:
        if file.startswith('.'): continue
        with open(os.path.join(root, file), "r") as auto:
            #if auto.name.endswith('DS_Store'): continue
            file_names.append(auto.name)

for file in file_names:
    with open(file, 'rb') as f:
        y, x = pickle.load(f)

        if x.shape != (128,128):
            print(x.shape)








quit()
#############
def get_file_names(path: str):
    file_names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file), "r") as auto:
                file_names.append(auto.name)
 

def instance_gettr(index: int):
    files = get_file_names(path)

    for file in files:
        print(file, type(file))
        with open(file, 'rb') as f:
            yield pickle.load(f)

g = instance_gettr('small_data_transformed')
y, x = next(g)
print(y)

quit()




#pickle.load()


######
quit()
path = 'data/LibriSpeech/train-clean-100'
contents = []
for i, (root, dirs, files) in enumerate(os.walk(path)):
    for file in files:
        if file.startswith('.'): continue
        with open(os.path.join(root, file), "r") as auto:
            contents.append(auto.name)
        break
    if i > 1: break

print(f'load: {contents[0]}')
test_sound, sr = librosa.load(contents[0])

S = librosa.feature.melspectrogram(y=test_sound, sr=sr, n_mels=128)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)

data = np.array_split(S_dB, 5, axis=1)

for i, img in enumerate(data):
    with open(f'test/file_name_{i}', 'wb') as f:
        pickle.dump((1,img),f)

quit()

img = cv2.resize(S_dB,(8,8))
np.save('test_features3',S_dB)
print(img.shape)
quit()

#np_img = np.array(img)
#print(np_img.shape)
#resize = np.resize(np_img, (1,1))
#print(resize.shape)
ax = img
plt.show()
quit()
quit()

fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
plt.show()
quit()



######





#directory = os.fsencode('data/LibriSpeech/train-clean-100')
path = 'data/LibriSpeech/train-clean-100'

contents = []
for root, dirs, files in os.walk(path):
     for file in files:
        with open(os.path.join(root, file), "r") as auto:
            print(auto)
            quit()



#test_file = contents[1]
#print(test_file)

contents.sort(reverse=True)
print(*contents, sep='\n')
print(len(contents))
quit()   
#for file in os.listdir(directory):
#    filename = os.fsdecode(file)
#    contents.append(filename)


#contents.remove('.DS_Store')
#contents = [int(x) for x in contents]
#contents.sort()

#for file in os.listdir()



     #if filename.endswith(".asm") or filename.endswith(".py"): 
         # print(os.path.join(directory, filename))
     #    continue
     #else:
     #    continue


quit()
contents = []
for i, x in enumerate(tf):
    print(x)
    contents.append(x)
    if i > 20: break

f=tf.extractfile(contents[1])
s = f.read().decode("utf-8") 
print(s[:10000])