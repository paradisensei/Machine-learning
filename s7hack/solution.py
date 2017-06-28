import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin_l_bfgs_b

# [1]
data = pd.read_csv('data/weights_heights.csv', usecols=['Height', 'Weight'])


# [2]
def plot_hist(feature, color, title, x, y):
    plt.figure()
    plt.hist(data[feature], color=color)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)

plot_hist('Height', 'blue', 'Height hist.', 'Height', 'Quantity')
plot_hist('Weight', 'green', 'Weight hist.', 'Weight', 'Quantity')
plt.show()


# [3.1]
data['BMI'] = np.round((data['Weight'] / data['Height'] ** 2) * 703, 5)


# [3.2]
def plot_relation(x, y, num):
    plt.subplot(320 + num)
    plt.plot(data[x], data[y], ',')
    plt.xlabel(x)
    plt.ylabel(y)

plt.figure(figsize=[14, 11], facecolor='w', dpi=65)
plot_relation('Weight', 'Height', 1)
plot_relation('Height', 'Weight', 2)
plot_relation('Weight', 'BMI', 3)
plot_relation('BMI', 'Weight', 4)
plot_relation('Height', 'BMI', 5)
plot_relation('BMI', 'Height', 6)
plt.show()


# [4]
def get_category(weight):
    if weight < 120:
        return 1
    elif weight >= 150:
        return 3
    else:
        return 2


data['weight_category'] = data['Weight'].apply(get_category)
data.boxplot(column='Height', by='weight_category')
plt.xlabel('Weight category')
plt.ylabel('Height')
plt.show()


# [5]
plt.scatter(data['Weight'], data['Height'], s=1, c='b')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()


# [6]
def hypothesis(w0, w1):
    return w0 + w1 * data['Weight']


def squared_error(w0, w1):
    sq_err = (data['Height'] - hypothesis(w0, w1)) ** 2
    return sum(sq_err)


# [7]
plt.scatter(data['Weight'], data['Height'], s=1, c='b')
plt.plot(data['Weight'], hypothesis(55, 0.04), ',')
plt.plot(data['Weight'], hypothesis(50, 0.22), ',')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()


# [8]
w1 = np.linspace(-5, 5, 100)
sq_err = []
w1_opt = 1000
sq_err_min = squared_error(50, w1_opt)
for w in w1:
    y = squared_error(50, w)
    if y < sq_err_min:
        sq_err_min = y
        w1_opt = w
    sq_err.append(y)
plt.plot(w1, sq_err)
plt.xlabel("W1")
plt.ylabel("Squared error function")
plt.show()


# [9]
print("Squared error min = %f at w0 = 50, w1 = %f" % (sq_err_min, w1_opt))
plt.scatter(data['Weight'], data['Height'], s=1)
plt.plot(data['Weight'], hypothesis(50, w1_opt), ',')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()


# [10]
w0 = np.linspace(-100, 100, 10)
w1 = np.linspace(-5, 5, 10)
sq_err = []
for i in w0:
    for j in w1:
        sq_err.append(squared_error(i, j))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
w0, w1 = np.meshgrid(w0, w1)
err = np.array([squared_error(i, j) for i, j in zip(np.ravel(w0), np.ravel(w1))]).reshape(w0.shape)
ax.plot_surface(w0, w1, err)
ax.set_xlabel("Intercept")
ax.set_ylabel("Slope")
ax.set_zlabel("Error")
plt.show()


# [11]
def sq_err(params, *args):
    w0 = params[0]
    w1 = params[1]
    weight = args[0]
    height = args[1]
    return sum((height - (w0 + w1 * weight)) ** 2)


res = fmin_l_bfgs_b(sq_err, x0=np.array([0, 0]), bounds=[(-100, 100), (-5, 5)],
                    args=(data['Weight'], data['Height']), approx_grad=True)
w = res[0]
print("Squared error min = %f at w0 = %f, w1 = %f" % (res[1], w[0], w[1]))
plt.scatter(data['Weight'], data['Height'], s=1)
plt.plot(data['Weight'], hypothesis(w[0], w[1]), ',')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()