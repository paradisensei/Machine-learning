import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

# custom imports
from nn import NeuralNetwork
from nn import Trainer


def test_performance(train_test):
    """
    :param train_test: an array-vector where
     train_test[0] - train_data,
     train_test[1] - train_target,
     train_test[2] - test_data,
     train_test[3] - test_target,
    """

    # k-NN
    __test_method(KNeighborsClassifier(), train_test)

    # decision tree
    __test_method(tree.DecisionTreeClassifier(), train_test)

    # random forest
    __test_method(RandomForestClassifier(), train_test)

    # sklearn built-in neural network implementation
    model = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5,
                            max_iter=1000, hidden_layer_sizes=(10, 7))
    __test_method(model, train_test)

    # custom neural network implementation with Sigmoid activation function
    __test_custom_nn(train_test)


def __test_method(model, train_test):
    i = 0
    attempts = 50

    # best_metrics: an array-vector where
    #  best_metrics[0] - best accuracy,
    #  best_metrics[1] - best precision,
    #  best_metrics[2] - best recall
    best_metrics = [0, 0, 0]

    while i < attempts:
        i += 1
        model.fit(train_test[0], train_test[1])
        prd = model.predict(train_test[2])
        best_metrics = __get_metrics(train_test[3], prd, best_metrics)

    __print_metrics(model.__class__.__name__, best_metrics)


def __test_custom_nn(train_test):
    i = 0
    attempts = 50

    # best_metrics: an array-vector where
    #  best_metrics[0] - best accuracy,
    #  best_metrics[1] - best precision,
    #  best_metrics[2] - best recall
    best_metrics = [0, 0, 0]

    # reshape target sets to match current network format
    __reshape(train_test)

    while i < attempts:
        i += 1
        nn = NeuralNetwork()
        t = Trainer(nn)
        t.train(train_test[0], train_test[1])
        prd = nn.forward(train_test[2])
        prd = map(lambda el: round(el), prd)
        best_metrics = __get_metrics(train_test[3], prd, best_metrics)

    __print_metrics('Custom neural network', best_metrics)


def __get_metrics(test_target, prd, best_metrics):
    accuracy = accuracy_score(test_target, prd)
    precision = precision_score(test_target, prd)
    recall = recall_score(test_target, prd)
    if accuracy > best_metrics[0]:
        best_metrics[0] = accuracy
    if precision > best_metrics[1]:
        best_metrics[1] = precision
    if recall > best_metrics[2]:
        best_metrics[2] = recall
    return best_metrics


def __print_metrics(method, metrics):
    print(method + ' : ')
    print(' accuracy : ' + str(metrics[0]))
    print(' precision : ' + str(metrics[1]))
    print(' recall : ' + str(metrics[2]))


def __reshape(train_test):
    train_target = train_test[1]
    test_target = train_test[3]
    train_test[1] = np.reshape(train_target, (train_target.shape[0], 1))
    train_test[3] = np.reshape(test_target, (test_target.shape[0], 1))
