# %%
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def single_neuron(x, w, b):
    z = x@w + b
    a = sigmoid(z)
    return a

def error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def derivative_error(y_true, y_pred):
    return -2 * (y_true - y_pred)

def delta(y_true, y_pred):
    return derivative_error(y_true, y_pred) * y_pred * (1 - y_pred)


# %%
# a
data = np.genfromtxt('2d_classification_multiple_neurons.csv', delimiter=',')
x_data = data[:, :2]
y_data = data[:, 2]
# print(x_data.shape)


plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap='bwr')
plt.title('2D classification data')
plt.show()


# %%
# 

batch_size = 10
learning_rate = 0.1
max_iterations = 20000
errors = []

N = 10

w1 = np.random.normal(size = (2, N))
b1 = np.random.normal(size = (1, N))
w2 = np.random.normal(size = (N, 1))
b2 = np.random.normal(size = (1, 1))

a = single_neuron(x_data, w1, b1)


plt.scatter(x_data[:, 0], x_data[:, 1], c=single_neuron(a, w2, b2).flatten(), cmap='bwr')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D classification data before training')
plt.show()


for iteration in range(max_iterations):
    rand_data_indecies = np.random.randint(len(x_data), size=batch_size)
    x_vals = x_data[rand_data_indecies]
    y_true = y_data[rand_data_indecies]
    neuron_vals = single_neuron(x_vals, w1, b1)
    y_pred = single_neuron(neuron_vals, w2, b2).flatten()


    err_vals = [error(y_true[i], y_pred[i]) for i in range(batch_size)]
    errors.append(np.mean(err_vals))
    
    dw2_vals = np.array([delta(y_true[i], y_pred[i])*neuron_vals[i] for i in range(batch_size)])
    db2_vals = np.array([delta(y_true[i], y_pred[i]) for i in range(batch_size)])

    w2 -= learning_rate * np.mean(dw2_vals, axis=0).reshape(w2.shape)
    b2 -= learning_rate * np.mean(db2_vals)


    dw1_vals = np.array([(delta(y_true[i], y_pred[i])*w2.T*neuron_vals[i]*(1 - neuron_vals[i])).T * x_vals[i].reshape(1, 2) for i in range(batch_size)])

    # db1_vals = np.array([(delta(y_true[i], y_pred[i])*w2.T*neuron_vals[i]*(1 - neuron_vals[i])).T for i in range(batch_size)])
    db1_vals = np.array([delta(y_true[i], y_pred[i])*neuron_vals[i]*(1 - neuron_vals[i]) for i in range(batch_size)])

    w1 -= learning_rate * np.mean(dw1_vals, axis=0).T
    b1 -= learning_rate * np.mean(db1_vals)



plt.plot(errors)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error over batches')
plt.show()


a = single_neuron(x_data, w1, b1)
plt.scatter(x_data[:, 0], x_data[:, 1], c=single_neuron(a, w2, b2).flatten(), cmap='bwr')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D classification data after training')
plt.show()
# %%
