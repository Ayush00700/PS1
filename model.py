import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dnn import L_layer_model
from predict import predict

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

inputs = pd.read_csv('data/input.csv')
# print(df.iloc[1:2])
# print(inputs)
# plt.plot(dataset)
outputs = pd.read_csv('data/output.csv')
output_label = pd.read_csv('data/output_label.csv')
# print(outputs)
together = pd.concat([inputs, output_label], axis=1)
togeth = together.to_numpy()
np.random.shuffle(togeth)
# print(inputs.shape)
# print(output_label.shape)
# print(outputs.shape)
X = inputs.to_numpy().T
Y = output_label.to_numpy().T
n_x, training_exs = X.shape
X_random = togeth[:, :216]
Y_random = togeth[:, 216, None]
X_random = X_random.T
Y_random = Y_random.T

for i in range(Y_random.shape[1]):
    if(Y_random[0, i] == -1):
        Y_random[0, i] = 0
# temp = list(zip(X, Y))
# X_rand, Y_rand = zip(*temp)
# X_random = np.asarray(X_rand)
# Y_random = np.asarray(Y_rand)
layers_dims = [n_x, 20, 10, 5, 1]  # 4-layer model
num_iterations = 1500


print(X.shape)
print(Y.shape)
# print(together.shape)
print(X_random.shape)
print(Y_random.shape)
# print(Y_random)
X_train = X_random[:, 0:30000]
X_test = X_random[:, 30000:36962]
Y_train = Y_random[:, 0:30000]
Y_test = Y_random[:, 30000:36962]
parameters, cost = L_layer_model(X_train, Y_train, layers_dims,
                                 num_iterations=num_iterations, learning_rate=0.0100, print_cost=True)
predictions_test = predict(X_test, Y_test, parameters)
# predictions_test = predict(X_test, Y_test, parameters)
# plot_costs(costs, learning_rate)
