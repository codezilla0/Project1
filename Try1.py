# -*- coding: utf-8 -*-
# copied from https://elitedatascience.com/imbalanced-classes

import pandas as pd
import numpy as np

df = pd.read_csv("balance-scale.data", names = ['balance', 'var1', 'var2', 'var3', 'var4'])
df.head()
df['balance'].value_counts()


df['balance'] = [1 if b == 'B' else 0 for b in df.balance]