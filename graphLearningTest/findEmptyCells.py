

import numpy as np
import pandas as pd
import pickle

file = open("../../ELMoPickled/ELMo_MRDA.pickle", "rb")
ELMo = pickle.load(file)

print(ELMo[~np.isnan(ELMo).any(axis = 1)].shape)