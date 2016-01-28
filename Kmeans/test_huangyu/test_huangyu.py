import pandas as pd
import numpy as np
import pickle
from new_gap_statistic import *

data = pickle.load(open('wallet_lv1.lv1','r'))
df = pd.DataFrame(data)
df1=df.fillna(0)
df2=df1
del df2[0]
ndata = df2.values
K,gaps = K_determin(ndata)
