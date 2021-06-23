import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dnn import L_layer_model
from predict import predict
import math
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
X_random = togeth[:, :216]
Y_random = togeth[:, 216, None]
X_random = X_random.T
Y_random = Y_random.T
n_x, training_exs = X_random.shape

for i in range(Y_random.shape[1]):
    if(Y_random[0, i] == -1):
        Y_random[0, i] = 0
layers_dims = [n_x, 20, 10, 5, 1]  # 4-layer model
num_iterations = 1500

print("Shape of X: {}".format(X_random.shape))
print("Shape of Y: {} ".format(Y_random.shape))
split_ratio = 8/10
partition_index = math.floor(X_random.shape[1]*split_ratio)
X_train = X_random[:, 0:partition_index]
X_test = X_random[:, partition_index:]
Y_train = Y_random[:, 0:partition_index]
Y_test = Y_random[:, partition_index:]
parameters, cost = L_layer_model(X_train, Y_train, layers_dims,
                                 num_iterations=num_iterations, learning_rate=0.1, print_cost=True)
predictions_test = predict(X_test, Y_test, parameters)
# plot_costs(costs, learning_rate)

count = 0
for i in range(Y_random.shape[1]):
    if(Y_random[0, i] == 0):
        count += 1
ratios = count/X_random.shape[1]
# print("Ratio of 0s: {}".format(ratios))
