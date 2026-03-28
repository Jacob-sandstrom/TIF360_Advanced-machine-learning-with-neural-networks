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

def derivative_weight(x, w, b, y_true, y_pred):
    return np.array([derivative_error(y_true, y_pred) * derivative_sigmoid(x@w + b) * x]).T

def derivative_bias(x, w, b, y_true, y_pred):
    return derivative_error(y_true, y_pred) * derivative_sigmoid(x@w + b)


# %%
# a
data = np.genfromtxt('2d_classification_single_neuron.csv', delimiter=',')
x_data = data[:, :2]
y_data = data[:, 2]
print(x_data.shape)

plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap='bwr')
plt.show()


# %%
# bc

batch_size = 10
learning_rate = 0.1
max_iterations = 10000
errors = []

w = np.random.normal(size = (2, 1))
b = np.random.normal()

plt.scatter(x_data[:, 0], x_data[:, 1], c=single_neuron(x_data, w, b).flatten(), cmap='bwr')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D classification data before training')
plt.show()


for iteration in range(max_iterations):
    rand_data_indecies = np.random.randint(len(x_data), size=batch_size)
    x_vals = x_data[rand_data_indecies]
    y_true = y_data[rand_data_indecies]
    y_pred = [single_neuron(x, w, b) for x in x_vals]
    
    err_vals = [error(y_true[i], y_pred[i]) for i in range(batch_size)]
    errors.append(np.mean(err_vals))
    
    dw_vals = [derivative_weight(x_vals[i], w, b, y_true[i], y_pred[i]) for i in range(batch_size)]
    db_vals = [derivative_bias(x_vals[i], w, b, y_true[i], y_pred[i]) for i in range(batch_size)]
    w -= learning_rate * np.mean(dw_vals, axis=0)
    b -= learning_rate * np.mean(db_vals)
    

print("Final weight:", w)
print("Final bias:", b)

plt.plot(errors)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error over batches')
plt.show()


plt.scatter(x_data[:, 0], x_data[:, 1], c=single_neuron(x_data, w, b).flatten(), cmap='bwr')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D classification data after training')
plt.show()
# %%
