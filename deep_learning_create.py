# %% [markdown]
# # Build my own deep learning functions
# ## Purpose
# Learn by practicing. Notations follow Andrew Ng's Coursera deep learning course.

# %% [markdown]
# ### Build the forward_prop function
# Note: If there is error while importing python modules while running this notebook in vscode, make sure the both the vscode python interpreter and ipython kernal are both set properly. 
# 
# ### Model structure
# The model consist of L-1 relu layers and one sigmoid layer.

# %%
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys

np.set_printoptions(edgeitems=4, linewidth=130)

np.random.seed(1) 
# TODO: with certain seeds, e.g. seed=1, the cost generates NAN

def plot_costs(costs, model_match_percents, learning_rate=0.01):
    x = np.arange(0, len(costs))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, costs, 'b*-')
    ax2.plot(x, model_match_percents, 'rx-')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost')
    ax2.set_ylabel('Match percentage (0 to 1)')
    ax1.yaxis.label.set_color('blue')
    ax2.yaxis.label.set_color('red')
    plt.show()

# %%
# Below function taken from course assignments
def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# %%
def relu(Z):
    '''
    Z: Input to the activate function

    A: Output of the relu function
    '''
    A = np.maximum(0, Z)
    assert(np.min(A) >= 0.0)
    return A

# %% [markdown]
# ### Initialization
# Initialize the parameters
# 
# ### Lesson learned
# When the weights are not initailized small, aka, without *0.01, the cost computation gives lots of NAN because the output are either too small or too large.

# %%
def model_init(layer_dims):
    parameters = {}
    for i in range(1, len(layer_dims)):
        parameters['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01 # the down-scaling is important
        parameters['b'+str(i)] = np.zeros((layer_dims[i], 1))
        # print(parameters['W'+str(i)].shape, parameters['b'+str(i)].shape)

    return parameters

# %%
def forward_prop(X, Y, parameters, classify_threshold = 0.5):
    '''
    # TODO: What's the row # variable?
    X: Input data. (n0???, m). n0: feature #; m: # of examples
    Y: Labels. (1, m)
    parameters: The model parameters

    cost: The return
    cache: The intermediate values
    predicted_result: The prediction given X
    model_match_percent: The matched percentage between prediction and Y
    '''
    L = len(parameters) // 2
    Al = X
    m = Y.shape[1]
    caches = {}
    for l in range(1, L):
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]

        assert(W.shape[0] == b.shape[0])
        Zl = np.dot(W, Al) + b
        Al = relu(Zl)
        caches['Z'+str(l)] = Zl
        caches['A'+str(l)] = Al

    ZL = np.dot(parameters['W'+str(L)], Al) + parameters['b'+str(L)]
    AL = 1 / (1 + np.exp(-ZL)) # sigmoid
    caches['Z'+str(L)] = ZL
    caches['A'+str(L)] = AL

    J = - np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T)
    cost = np.squeeze(np.sum(J)) / m
    assert(cost.shape == ())

    predicted_result = np.ones_like(AL) * (AL > classify_threshold)
    model_match_percent = 1 - np.sum(np.logical_xor(predicted_result, Y)) / m

    return cost, caches, predicted_result, model_match_percent

# %% [markdown]
# ### Back prop
# The back propogation is to calculate the partial derivatives of the cost w.r.t. all the weights and biases. So that the learning algorithem can nudge the weights and biases to reduce the cost by a tiny bit. 
# 
# #### Notations
# Following cousera's notation, Z1, Z2, etc. are the linear combination of the input with biases for layers 1, 2, and so on. A1, A2, etc. are the activation outputs for layers 1, 2, and so on.
# 
# $dZ_1=\frac{d cost}{d Z1}$

# %%
def back_prop(X, Y, caches):
    '''
    X: Input data. (n0???, m). n0: feature #; m: # of examples
    Y: Labels. (1, m)
    parameters: The model parameters

    grads: return the gradients
    '''
    grads = {}
    m = Y.shape[1]
    L = len(caches) // 2
    caches['A0'] = X

    # The last layer is a sigmoid layer
    grads['dZ' + str(L)] = 1/m * (caches['A' + str(L)] - Y)
    grads['dW' + str(L)] = np.dot(grads['dZ' + str(L)], caches['A' + str(L-1)].T) ## Lesson learned, use previous layer Activation output
    grads['db' + str(L)] = np.sum(grads['dZ' + str(L)], axis=1, keepdims=True)

    for l in range(L-1, 0, -1):
        # The remaining layers are RELU layers
        grads['dZ' + str(l)] = np.ones_like(caches['Z' + str(l)]) * (caches['Z' + str(l)] > 0.0)
        grads['dW' + str(l)] = np.dot(grads['dZ' + str(l)], caches['A' + str(l-1)].T)
        grads['db' + str(l)] = np.sum(grads['dZ' + str(l)], axis=1, keepdims=True)

    return grads

# %% [markdown]
# ### The overall model
# Here is the overall learning model with hyper-parameters

# %%
def train_model(X, Y, layer_dims, number_of_iterations = 5, learning_rate = 0.01):
    # model initialization
    parameters = model_init(layer_dims)

    # results
    costs = np.zeros((number_of_iterations))
    model_match_percents = np.zeros((number_of_iterations))

    for i in range(number_of_iterations):
        costs[i], caches, predicted_result, model_match_percents[i] = forward_prop(X, Y, parameters)
        # print('len(train_set_y_orig)=', len(train_set_y_orig))
        # print('train_set_y_orig.shape', train_set_y_orig.shape)
        if i % 100 == 0:
            print('Iteration {0:5.0f}, cost={1:.6f}, prediction accuracy={2:.4f}'.format(i, costs[i], model_match_percents[i]))
        
        grads = back_prop(X, Y, caches)

        for l in range(1, len(layer_dims)):
            parameters['W'+str(l)] = parameters['W'+str(l)] - grads['dW'+str(l)] * learning_rate
            parameters['b'+str(l)] = parameters['b'+str(l)] - grads['db'+str(l)] * learning_rate
            # print('Parameter updated: ', 'W'+str(l))

    plot_costs(costs, model_match_percents, learning_rate)

    return parameters, costs

def main():
    # load data and pre-processing
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    # print('train_set_x_orig.shape=', train_set_x_orig.shape)
    # plt.imshow(train_set_x_orig[7])
    # plt.show()

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    # print('train_set_x_flatten.shape=', train_set_x_flatten.shape)
    # print('test_set_x_flatten.shape=', test_set_x_flatten.shape)
    # print(train_set_x_flatten)
    # print(train_set_x_flatten.max())

    train_set_x = train_set_x_flatten / 255.0
    test_set_x = test_set_x_flatten / 255.0
    layer_dims = [train_set_x.shape[0], 20, 7, 5, 1]
    learning_rate = 0.005
    model, costs = train_model(train_set_x, train_set_y_orig, layer_dims, 600, learning_rate)

    # predict(model, test_set_x_orig, test_set_y_orig)

if __name__ == "__main__":
    main()

# %% Temporary test code
# for i in range(2, 0, -1):
#     print('i=', i)
# sys.exit()

