import pandas as pd
import numpy as np

train = pd.read_excel('train.xlsx', header=0)
test = pd.read_excel('test.xlsx', header=0)
train = np.array(train)
test = np.array(test)