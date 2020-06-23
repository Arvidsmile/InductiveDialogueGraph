
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

file = open("../../ELMoPickled/ELMo_MSDialog.pickle", "rb")
ELMo = pickle.load(file)

df = pd.read_csv("../../CSVData/MSDialog.csv")
print(df[df.duplicated(['Utterance'], keep = False)]['Utterance'].value_counts())

print("-" * 100)
# print(df[df['Utterance'] == 'Uh-huh .'])
print("-" * 100)
# print(df["Dialogue ID"].value_counts())
# print("-" * 100)
print(df.shape, ELMo.shape)
# print(ELMo[1242], df['Utterance'].loc[1242])
# print(ELMo[1243], df['Utterance'].loc[1243])
print("-" * 100)

cossim = cosine_similarity(ELMo)
np.fill_diagonal(cossim, 0)
indeces = np.where(cossim == cossim.max())
indeces = np.where(np.logical_and(cossim >= 0.98, cossim < 0.99))
x_y_coords = list(zip(indeces[0], indeces[1]))
print(x_y_coords)
print(cossim[x_y_coords[0]])
print(df.loc[x_y_coords[0][0]], df['Utterance'].loc[x_y_coords[4][0]])
print(df.loc[x_y_coords[0][1]], df['Utterance'].loc[x_y_coords[4][1]])

