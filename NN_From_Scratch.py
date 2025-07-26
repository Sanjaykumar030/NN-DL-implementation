import numpy as np

#Activation Functions - 

#Sigmoid Function - Binary Classification
def sigmoid(Z):
    return 1/(1+ np.exp(-Z))

#ReLU Function - Outputs the input if positive, otherwise zero
def relu(Z):
    return np.maximum(0, Z)

def tanh(Z):
    return np.tanh(Z)  # Hyperbolic tangent function


#Sigmoid Derivative - For BackPropagation 
def sigmoid_derivative(A):
    return A * (1- A)
#ReLu Derivative - For BackPropagation
def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)

#Tanh Derivative - For BackPropagation
def tanh_derivative(A):
    return 1 - np.square(A)

#Initializing the weights and biases
def initialize_parameters(n_x, n_h, n_y): #n_x - Number of input features, n_h - Number of hidden neurons, n_y - Number of output neurons
    np.random.seed(1) # For reproducibility
    W1 = np.random.randn(n_h, n_x) * 0.01  # Weight matrix for input to hidden layer #0.01 for small initial weights
    b1 = np.zeros((n_h, 1))  # Bias for hidden layer
    W2 = np.random.randn(n_y, n_h) * 0.01  # Weight matrix for hidden to output layer
    b2 = np.zeros((n_y, 1))  # Bias for output layer
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

#Forward Propagation - Used to compute the output of the network
def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    Z1 = np.dot(W1, X) + b1  # Linear transformation for hidden layer
    A1 = tanh(Z1)  # Activation for hidden layer
    Z2 = np.dot(W2, A1) + b2  # Linear transformation for output layer
    A2 = sigmoid(Z2)  # Activation for output layer

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}  # Store intermediate values for backpropagation
    return A2, cache

#Computing the cost
def compute_cost(A2, Y):
    m = Y.shape[1]  # Number of examples
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m  # Cross-entropy cost
    return np.squeeze(cost)  # Ensure cost is a scalar

#Backward Propagation - Used to update weights and biases (Derivatives)
def backward_propagation(X, Y, parameters, cache):
    m = X.shape[1]  # Number of examples
    W2 = parameters["W2"]
    A1, A2 = cache["A1"], cache["A2"]
    Z1 = cache["Z1"]    

    dZ2 = A2 - Y  # Derivative of cost with respect to Z2
    dW2 = (1/m) * np.dot(dZ2, A1.T)  # Gradient for W2
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)  # Gradient for b2

    dZ1 = np.dot(W2.T, dZ2) * tanh_derivative(Z1)  # Derivative of cost with respect to Z1
    dW1 = (1/m) * np.dot(dZ1, X.T)  # Gradient for W1
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)  # Gradient for b1

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads # Return gradients for updating parameters

#Updating parameters - Used to adjust weights and biases based on gradients
def update_parameters(parameters, grads, learning_rate):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    return parameters   

#Model Function
def model(X, Y, n_h, num_iterations=10000, learning_rate=0.01, print_cost=False):
    n_x = X.shape[0]  # Number of input features
    n_y = Y.shape[0]  # Number of output neurons

    parameters = initialize_parameters(n_x, n_h, n_y)  # Initialize weights and biases

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(X, Y, parameters, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0: # Print cost every 1000 iterations
            print(f"Cost after iteration {i}: {cost:.4f}")
    return parameters # Return trained parameters

#Prediction
def predict(X, parameters):
    A2, _ = forward_propagation(X, parameters)
    return A2 > 0.5

#Sample Dataset (Dummy Data)

if __name__ == "__main__":
    np.random.seed(1)
    X = np.random.randn(2, 500) # 2 features, 500 examples
    Y = (np.sum(X, axis=0) > 0).astype(int).reshape(1, -1) #Sample boundary condition

    params = model(X, Y, n_h=4, num_iterations=5000, learning_rate=0.01, print_cost=True)
    predictions = predict(X, params)
    accuracy = np.mean(predictions == Y) * 100 # Calculate accuracy
    print(f"Training accuracy: {accuracy:.2f}%")

