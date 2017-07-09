import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from utils import normalize
from utils import complement

# read data
data = pd.read_csv('data/train.csv', sep=';')
data = data.drop(['id', 'smoke', 'alco', 'active'], axis=1)
test_data = pd.read_csv('data/test.csv', sep=';')
test_data = test_data.drop(['id', 'smoke', 'alco', 'active'], axis=1)

# normalize some features
normalize(data, ('age', 'height', 'weight', 'ap_hi', 'ap_lo'))
normalize(test_data, ('age', 'height', 'weight', 'ap_hi', 'ap_lo'))

# fill absent features
# complement(test_data, ('smoke', 'alco', 'active'))

# split for data and target
X = data.drop('cardio', axis=1)
y = data['cardio']

# convert to numpy arrays
X = np.asarray(X, dtype='|S6').astype(np.float)
y = np.asarray(y, dtype='|S6').astype(np.int)

# train
# model = LogisticRegression(solver='newton-cg')
model = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5,
                            max_iter=1000, hidden_layer_sizes=(10, 8))
model.fit(X, y)

# predict
prd = model.predict_proba(test_data)
prd = np.delete(prd, 0, axis=1)

# write prediction
prd = pd.DataFrame(data=prd)
prd.to_csv('res/y_test.csv', index=None, header=None)