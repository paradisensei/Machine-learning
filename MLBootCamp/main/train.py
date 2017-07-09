import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

from utils import normalize

# read data
data = pd.read_csv('data/train.csv', sep=';')
data = data.drop(['id', 'smoke', 'alco', 'active'], axis=1)

# normalize some features
normalize(data, ('age', 'height', 'weight', 'ap_hi', 'ap_lo'))

# split for data and target
X = data.drop('cardio', axis=1)
y = data['cardio']

# convert to numpy arrays
X = np.asarray(X, dtype='|S6').astype(np.float)
y = np.asarray(y, dtype='|S6').astype(np.int)

# split data to train and test sets
m = X.shape[0]
test_idx = range(0, m, 8)
train_X = np.delete(X, test_idx, axis=0)
train_y = np.delete(y, test_idx)
test_X = X[test_idx]
test_y = y[test_idx]

# train
# solver = newton-cg, liblinear, lbfgs, sag
# model = LogisticRegression(solver='lbfgs')
# solver = lbfgs, sgd, adam
model = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5,
                            max_iter=1000, hidden_layer_sizes=(10, 8))
model.fit(train_X, train_y)

# predict
prd = model.predict(test_X)
prd_proba = model.predict_proba(test_X)

# estimate model quality
print 'accuracy = %f' % accuracy_score(test_y, prd)
print 'log. loss = %f' % log_loss(test_y, prd_proba)