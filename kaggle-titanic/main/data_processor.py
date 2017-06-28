import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


def process():
    # read data and get rid of useless features
    data = pd.read_csv('../data/train.csv')
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    # add 'female' feature based on 'sex' feature
    data['Female'] = data['Sex'].apply(__get_female)
    data.drop(['Sex'], axis=1, inplace=True)

    # add 'family' feature based on 'sibsp' & 'parch' features
    __add_family(data)

    # fill 'age' feature where it is absent
    __fill_nan_age(data)

    return data


def normalize(data):
    # normalize 'pclass' feature
    data['Pclass'] = data['Pclass'].apply(__normalize_class)
    # normalize 'age' feature
    data['Age'] = data['Age'].apply(__normalize_age)
    # normalize 'fare' feature
    data['Fare'] = data['Fare'].apply(__normalize_fare)


# helper functions
def __get_female(sex):
    if sex == 'male':
        return 0
    else:
        return 1


def __add_family(data):
    # add 'family' feature
    data['Family'] = data['Parch'] + data['SibSp']
    data['Family'].loc[data['Family'] > 0] = 1
    data['Family'].loc[data['Family'] == 0] = 0

    # drop 'sibsp' & 'parch' features
    data.drop(['SibSp', 'Parch'], axis=1, inplace=True)


def __fill_nan_age(data):
    # get average, std, and number of NaN values in train
    average_age_titanic = data['Age'].mean()
    std_age_titanic = data['Age'].std()
    count_nan_age_titanic = data['Age'].isnull().sum()

    # generate random numbers between (mean - std) & (mean + std)
    rand = np.random.randint(average_age_titanic - std_age_titanic,
                             average_age_titanic + std_age_titanic,
                             size=count_nan_age_titanic)

    # fill NaN values in Age column with random values generated
    data['Age'][np.isnan(data['Age'])] = rand


def __normalize_class(p_class):
    return p_class / float(3)


def __normalize_age(age):
    return age / float(80)


def __normalize_fare(fare):
    return fare / 512.3292
