
def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    params -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = params.copy()
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    # (≈ 2 lines of code)
    for l in range(L):
        # YOUR CODE STARTS HERE
        parameters["W" + str(l+1)] -= learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate*grads["db"+str(l+1)]
        # YOUR CODE ENDS HERE
    return parameters
