# %%
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def single_neuron(x, w, b):
    z = x@w + b
    # a = sigmoid(z)
    a = np.tanh(z)
    return a

def error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def derivative_error(y_true, y_pred):
    return -2 * (y_true - y_pred)

# def delta(y_true, y_pred):
#     # return derivative_error(y_true, y_pred) * y_pred * (1 - y_pred)
#     return derivative_error(y_true, y_pred) * (1 - y_pred ** 2)


# %%
# a
data = np.genfromtxt('function_approximation.csv', delimiter=',')
x_data = data[:, 0]
y_data = data[:, 1]
# print(x_data.shape)


plt.scatter(x_data, y_data, c='b', marker='o')
plt.title('Function approximation data')
plt.show()


# %%
# 

batch_size = 10
learning_rate = 0.1
max_iterations = 10000
errors = []

N = 10
M = 8

sizes = [1, N, M, 1]

biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]



for iteration in range(max_iterations):
    rand_data_indecies = np.random.randint(len(x_data), size=batch_size)
    x_vals = x_data[rand_data_indecies]
    y_true = y_data[rand_data_indecies]

    weight_derivatives = [np.zeros(w.shape) for w in weights]
    bias_derivatives = [np.zeros(b.shape) for b in biases]

    err = 0
    for i in range(batch_size):
        x = x_vals[i]
        y_t = y_true[i]

        z1 = weights[0]* x + biases[0]
        a1 = np.tanh(z1)

        z2 = np.dot(weights[1], a1) + biases[1]
        a2 = np.tanh(z2)

        z3 = np.dot(weights[2], a2) + biases[2]
        y_pred = np.tanh(z3)


        delta = derivative_error(y_t, y_pred) * (1 - y_pred**2)
        weight_derivatives[2] += np.dot(delta, a2.T)
        bias_derivatives[2] += delta

        delta = np.dot(weights[2].T, delta) * (1 - a2**2)
        weight_derivatives[1] += np.dot(delta, a1.T)
        bias_derivatives[1] += delta

        delta = np.dot(weights[1].T, delta) * (1 - a1**2)
        weight_derivatives[0] += delta* x
        bias_derivatives[0] += delta




        err += error(y_t, y_pred)[0][0]


    

    errors.append(err / batch_size)

    for j in range(len(weights)):
        weights[j] -= learning_rate * weight_derivatives[j] / batch_size
        biases[j] -= learning_rate * bias_derivatives[j] / batch_size



plt.plot(errors)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error over batches')
plt.show()

y = []
for i in range(len(x_data)):
    x = x_data[i]

    z1 = weights[0]* x + biases[0]
    a1 = np.tanh(z1)

    z2 = np.dot(weights[1], a1) + biases[1]
    a2 = np.tanh(z2)

    z3 = np.dot(weights[2], a2) + biases[2]
    y_pred = np.tanh(z3)
    y.append(y_pred[0][0])
# a = [single_neuron([x_data[i]], w1, b1) )]
print(y)

plt.scatter(x_data, y, color='blue', label='Predicted')
plt.scatter(x_data, y_data, color='red', label='True')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function approximation after training')
plt.show()
# %%
