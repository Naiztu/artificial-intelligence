# %% [markdown]
# # Activity 2.1 Generation or selection of the data set
# 

# %% [markdown]
# **José Ángel Rico Mendieta - A01707404**

# %% [markdown]
# ## Get, generate or increase a data set

# %% [markdown]
# ### Data set selected - Music features
# #### About Dataset
# ##### Context
# A music genre is a conventional category that identifies pieces of music as belonging to a shared tradition or set of conventions. It is to be distinguished from musical form and musical style. The features extracted from these waves can help the machine distinguish between them.
# 
# ##### Content
# The features in this dataset are extracted from the dataset provided [here](http://marsyas.info/) which consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format. The code used to extract features is at this GitHub repo. Features are extracted using [libROSA](https://librosa.github.io/librosa/) library.
# 
# ##### Acknowledgements
# The credits to this dataset go to [MARSYAS](http://marsyas.info/).
# 
# ##### Inspiration
# Due to the artistic nature of music, the classifications are often arbitrary and controversial, and some genres may overlap. Train a model and know to which genre your favourite piece of music belong to.

# %%
import pandas as pd

# %%

ruta = './'
df = pd.read_csv(ruta+'data.csv')

# %%
print(df)

# %%
print(df.describe())

# %%
print(df.groupby('label').count())

# %%
import matplotlib.pyplot as plt

labels = df['label'].unique()
counts = df.groupby('label').count().iloc[:, 0].values

plt.bar(labels, counts)

plt.title('Number of Labels')
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.xticks(rotation=90)

plt.show()

# %% [markdown]
# ## Make the separation of test and training sets.
# 

# %%
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.1)

# %%
print(train_df)

# %%
print(test_df)

# %%
print(val_df)