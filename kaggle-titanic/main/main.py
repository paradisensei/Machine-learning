import numpy as np

# custom imports
import data_processor
import test

# get and process the data
data = data_processor.process()

# normalize 'pclass', 'age', 'fare' features within a range of [0, 1]
data_processor.normalize(data)

# split for data and target
target = data['Survived']
data = data.drop('Survived', axis=1)

# convert to numpy arrays
data = np.asarray(data, dtype='|S6').astype(np.float)
target = np.asarray(target, dtype='|S6').astype(np.float)

# split data to train and test sets
m = data.shape[0]
test_idx = range(0, m, 8)
train_data = np.delete(data, test_idx, axis=0)
train_target = np.delete(target, test_idx)
test_data = data[test_idx]
test_target = target[test_idx]

# test performance of different algorithms
test.test_performance([train_data, train_target, test_data, test_target])