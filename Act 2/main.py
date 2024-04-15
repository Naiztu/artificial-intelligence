# %% [markdown]
# # Activity 2.2 Data preprocessing

# %% [markdown]
# **José Ángel Rico Mendieta - A01707404**

# %%
import pandas as pd

# %%
ruta = './'
df = pd.read_csv(ruta+'data.csv')

# %% [markdown]
# ## Data preprocessing

# %%
map_labels = {'blues':0, 'classical':1, 'country':2, 'disco':3, 'hiphop':4, 
              'jazz':5, 'metal':6, 'pop':7, 'reggae':8, 'rock':9 }

df['label'] = df['label'].replace(map_labels)
df = df.drop(['filename'], axis=1)
print(df)

# %% [markdown]
# ## Apply escalation techniques.

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# %%
print(scaled_df)

# %%
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(scaled_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.1)

# %%
print(train_df)

# %%
print(test_df)

# %%
print(val_df)