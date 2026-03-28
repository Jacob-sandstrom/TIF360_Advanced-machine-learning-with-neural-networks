# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# a
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def single_neuron(x, w, b):
    z = w * x + b
    a = sigmoid(z)
    return a

def error(y_true, y_pred):
    return (y_true - y_pred) ** 2

# %%
# b
w = np.random.normal()
b = np.random.normal()

print("Initial weight:", w)
print("Initial bias:", b)

data = np.genfromtxt('1d_classification_single_neuron.csv', delimiter=',')
x_data = data[:, 0]
y_data = data[:, 1]

plt.scatter(x_data, single_neuron(x_data, w, b), color='blue', label='Predicted')
plt.scatter(x_data, y_data, color='red', label='True')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('1D Classification Data')
plt.show()


# %%
# c
def derivative_error(y_true, y_pred):
    return -2 * (y_true - y_pred)

def derivative_weight(x, w, b, y_true, y_pred):
    return derivative_error(y_true, y_pred) * derivative_sigmoid(w * x + b) * x

def derivative_bias(x, w, b, y_true, y_pred):
    return derivative_error(y_true, y_pred) * derivative_sigmoid(w * x + b)


def update_parameters(w, b, x, y_true, y_pred, learning_rate):
    dw = derivative_weight(x, w, b, y_true, y_pred)
    db = derivative_bias(x, w, b, y_true, y_pred)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# %%
# d
learning_rate = 0.01
max_iterations = 100000
errors = []


w = np.random.normal()
b = np.random.normal()

print("Initial weight:", w)
print("Initial bias:", b)


# print(x_data[])
for iteration in range(max_iterations):
    rand_data_index = np.random.randint(len(x_data))
    x = x_data[rand_data_index]
    y_true = y_data[rand_data_index]
    y_pred = single_neuron(x, w, b)
    
    err = error(y_true, y_pred)
    errors.append(err)

    w, b = update_parameters(w, b, x, y_true, y_pred, learning_rate)



print("Final weight:", w)
print("Final bias:", b)

plt.plot(errors)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error over Iterations')
plt.show()

plt.scatter(x_data, single_neuron(x_data, w, b), color='blue', label='Predicted')
plt.scatter(x_data, y_data, color='red', label='True')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('1D Classification Data After Training')
plt.show()



# %%
# e

batch_size = 10
learning_rate = 0.1
max_iterations = 10000
errors = []


w = np.random.normal()
b = np.random.normal()

print("Initial weight:", w)
print("Initial bias:", b)


# print(x_data[])
for iteration in range(max_iterations):
    rand_data_indecies = np.random.randint(len(x_data), size=batch_size)
    x_vals = x_data[rand_data_indecies]
    y_true = y_data[rand_data_indecies]
    y_pred = [single_neuron(x, w, b) for x in x_vals]
    
    # err_vals = error(y_true, y_pred)
    err_vals = [error(y_true[i], y_pred[i]) for i in range(batch_size)]
    errors.append(np.mean(err_vals))

    # w, b = update_parameters(w, b, x_vals, y_true, y_pred, learning_rate)
    dw_vals = [derivative_weight(x_vals[i], w, b, y_true[i], y_pred[i]) for i in range(batch_size)]
    db_vals = [derivative_bias(x_vals[i], w, b, y_true[i], y_pred[i]) for i in range(batch_size)]
    w -= learning_rate * np.mean(dw_vals)
    b -= learning_rate * np.mean(db_vals)
    


print("Final weight:", w)
print("Final bias:", b)

plt.plot(errors)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error over batches')
plt.show()

plt.scatter(x_data, single_neuron(x_data, w, b), color='blue', label='Predicted')
plt.scatter(x_data, y_data, color='red', label='True')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('1D Classification Data After Training')
plt.show()
# %%
