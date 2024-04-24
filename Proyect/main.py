# %%
import librosa
import numpy as np
import pandas as pd
import os
import csv
import time

# %%
model_name = 'model-multi'

# %%
test = pd.DataFrame({}, columns=[ 'tempo', 'beats', 'chroma_stft', 'rmse',
       'spectral_centroid', 'spectral_bandwidth', 'rolloff',
       'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
       'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
       'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',
       'mfcc20'])
for filename in os.listdir(f'./music'):
    songname = f'./music/{filename}'
    y, sr = librosa.load(songname, mono=True, duration=30)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(y=y, sr=sr)[0]
    beats = librosa.feature.tempo(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{tempo} {beats.shape[0]} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    
    test.loc[len(test)] = to_append.split()  

test = test.astype('float64')
test 

# %%
import tensorflow as tf

# %%
test

# %%
model = tf.keras.models.load_model(model_name+'.h5')

predicted = model.predict(test)
np.argmax(predicted, axis=1)

# %%
mapper = {0:'blues', 1:'classical', 2:'country', 3:'disco', 4:'hiphop', 5:'jazz', 6:'metal', 7:'pop', 8:'reggae', 9:'rock'}
mapped = [mapper[i] for i in np.argmax(predicted, axis=1)]
print(mapped)


