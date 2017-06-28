import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

# read data
data = pd.read_csv('data/train.csv', sep=';')
data = data.drop(['smoke', 'alco', 'active'], axis=1)
test_x = pd.read_csv('data/test.csv', sep=';')
test_x = test_x.drop(['smoke', 'alco', 'active'], axis=1)

# convert data to numpy arrays
data = np.asarray(data, dtype='|S6').astype(np.float)
test_x = np.asarray(test_x, dtype='|S6').astype(np.float)

# split
train_x = np.delete(data, 9, axis=1)
train_y = data[:, 9]

# train
model = LogisticRegression()
model.fit(train_x, train_y)

# predict
prd = model.predict_proba(test_x)
prd = np.delete(prd, 0, axis=1)

# write prediction
prd = pd.DataFrame(data=prd)
prd.to_csv('res/y_test.csv', index=None, header=None)