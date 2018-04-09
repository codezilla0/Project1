# -*- coding: utf-8 -*-
# copied from https://elitedatascience.com/imbalanced-classes

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("balance-scale.data", names = ['balance', 'var1', 'var2', 'var3', 'var4'])
df.head()
df['balance'].value_counts()


df['balance'] = [1 if b == 'B' else 0 for b in df.balance]

type(df.balance)
df.balance.value_counts()
type(df['balance'])

y = df.balance

X = df.drop('balance', axis=1)

clf_0 = LogisticRegression().fit(X, y)

pred_clf_0 = clf_0.predict(X)

sum(y - pred_clf_0)