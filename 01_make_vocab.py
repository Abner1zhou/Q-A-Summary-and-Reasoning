import pandas as pd
from utils.config import *

merger_df = pd.read_csv(merger_seg_path, header=0, names=['a','b'])

vocab = set(' '.join(merger_df['b']).split(' '))

with open('database/vocab.txt', 'w') as f:
    for i in vocab:
        f.write(i+'\n')