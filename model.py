import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dnn import L_layer_model
from L_model_forward import L_model_forward
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

# print(inputs.shape)
# print(output_label.shape)
# print(outputs.shape)
X = inputs.to_numpy().T
Y = output_label.to_numpy().T
for i in range(Y.shape[1]):
    if(Y[0, i] == -1):
        Y[0, i] = 0
X_train = X[:, 0:30000]
X_test = X[:, 30000:36962]
Y_train = Y[:, 0:30000]
Y_test = Y[:, 30000:36962]
n_x, training_exs = X.shape
layers_dims = [n_x, 20, 10, 5, 1]  # 4-layer model
num_iterations = 1500
parameters, cost = L_layer_model(X_train, Y_train, layers_dims,
                                 num_iterations=num_iterations, learning_rate=0.0070, print_cost=True)
predictions_test = predict(X_test, Y_test, parameters)
# print(parameters)
# plot_costs(costs, learning_rate)
# predictions_test = predict(test_x, test_y, parameters)
