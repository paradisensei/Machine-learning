import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


def normalize(data, features):
    for feature in features:
        mean = data[feature].mean()
        std = data[feature].std()
        data[feature] = data[feature].apply(__normalize_feature, args=(mean, std))


def complement(data, features):
    for feature in features:
        mean = data[feature].mean()
        std = data[feature].std()
        count_null_feature = data[feature].isnull().sum()

        # generate random numbers between (mean - std) & (mean + std)
        rand = np.random.randint(mean - std, mean + std, size=count_null_feature)

        # fill null values in 'feature' column with random values generated
        data[feature][np.isnan(data[feature])] = rand


def __normalize_feature(feature, mean, std):
    return (feature - mean) / std
