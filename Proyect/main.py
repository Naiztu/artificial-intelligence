# %%
import librosa
import numpy as np
import pandas as pd
import os

# %%
model_name = 'model'

# %%
test = pd.DataFrame({}, columns=['name', 'tempo', 'beats', 'chroma_stft', 'rmse',
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
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{filename} {tempo} {beats.shape[0]} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    
    test.loc[len(test)] = to_append.split()  

test = test.astype({
    'tempo': 'float64',
    'beats': 'float64',
    'chroma_stft': 'float64',
    'rmse': 'float64',
    'spectral_centroid': 'float64',
    'spectral_bandwidth': 'float64',
    'rolloff': 'float64',
    'zero_crossing_rate': 'float64',
    'mfcc1': 'float64',
    'mfcc2': 'float64',
    'mfcc3': 'float64',
    'mfcc4': 'float64',
    'mfcc5': 'float64',
    'mfcc6': 'float64',
    'mfcc7': 'float64',
    'mfcc8': 'float64',
    'mfcc9': 'float64',
    'mfcc10': 'float64',
    'mfcc11': 'float64',
    'mfcc12': 'float64',
    'mfcc13': 'float64',
    'mfcc14': 'float64',
    'mfcc15': 'float64',
    'mfcc16': 'float64',
    'mfcc17': 'float64',
    'mfcc18': 'float64',
    'mfcc19': 'float64',
    'mfcc20': 'float64'
})

# %%
import tensorflow as tf

# %%
test

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = pd.DataFrame(scaler.fit_transform(test.drop('name', axis=1)), columns=['tempo', 'beats', 'chroma_stft', 'rmse', 'spectral_centroid',
       'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2',
       'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
       'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17',
       'mfcc18', 'mfcc19', 'mfcc20'])

# %%
scaled_df

# %%
model = tf.keras.models.load_model(model_name+'.h5')

predicted = model.predict(scaled_df)
np.argmax(predicted, axis=1)

# %%
#mapper = {0:'blues', 1:'classical', 2:'country', 3:'disco', 4:'hiphop', 5:'jazz', 6:'metal', 7:'pop', 8:'reggae', 9:'rock'}
mapper = {0:'pop', 1:'classical'}

mapped = [mapper[i] for i in np.argmax(predicted, axis=1)]
print(mapped)


