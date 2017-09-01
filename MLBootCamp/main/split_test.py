import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier

from utils import normalize

# read data
data = pd.read_csv('data/train.csv', sep=';')
data_1 = data.drop(['id'], axis=1)
data_2 = data_1.drop(['smoke', 'alco', 'active'], axis=1)
test_data = pd.read_csv('data/test.csv', sep=';')
test_data = test_data.drop(['id'], axis=1)

# normalize some features
normalize(data_1, ('age', 'height', 'weight', 'ap_hi', 'ap_lo'))
normalize(data_2, ('age', 'height', 'weight', 'ap_hi', 'ap_lo'))
normalize(test_data, ('age', 'height', 'weight', 'ap_hi', 'ap_lo'))

test_data_smoke_nan = test_data['smoke'].isnull()
test_data_alco_nan = test_data['alco'].isnull()
test_data_active_nan = test_data['active'].isnull()

idx = []
nan_idx = []
for i in range(0, len(test_data)):
    if test_data_smoke_nan[i] or test_data_alco_nan[i] or test_data_active_nan[i]:
        nan_idx.append(i)
    else:
        idx.append(i)

test_data_1 = test_data.drop(nan_idx)
test_data_2 = test_data.ix[nan_idx]
test_data_2 = test_data_2.drop(['smoke', 'alco', 'active'], axis=1)

# split for data and target
X_1 = data_1.drop('cardio', axis=1)
X_2 = data_2.drop('cardio', axis=1)
y = data['cardio']

# convert to numpy arrays
X_1 = np.asarray(X_1, dtype='|S6').astype(np.float)
X_2 = np.asarray(X_2, dtype='|S6').astype(np.float)
y = np.asarray(y, dtype='|S6').astype(np.int)

# train
# model = LogisticRegression(solver='newton-cg')
model_1 = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5,
                            max_iter=1000, hidden_layer_sizes=(10, 8))
model_1.fit(X_1, y)

model_2 = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5,
                            max_iter=1000, hidden_layer_sizes=(10, 8))
model_2.fit(X_2, y)

# predict
prd_1 = model_1.predict_proba(test_data_1)
prd_1 = np.delete(prd_1, 0, axis=1)

prd_2 = model_2.predict_proba(test_data_2)
prd_2 = np.delete(prd_2, 0, axis=1)

prd = []

n = len(idx)
m = len(nan_idx)
i = j = 0

while i < n and j < m:
    if idx[i] < nan_idx[j]:
        prd.append(prd_1[i])
        i += 1
    else:
        prd.append(prd_2[j])
        j += 1

while i < n:
    prd.append(prd_1[i])
    i += 1

while j < m:
    prd.append(prd_2[j])
    j += 1

# write prediction
prd = pd.DataFrame(data=prd)
prd.to_csv('res/y_test.csv', index=None, header=None)