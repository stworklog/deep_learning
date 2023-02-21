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
from datetime import datetime
import pickle
import signal

global_stop_flag = False

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    global global_stop_flag 
    global_stop_flag = True

signal.signal(signal.SIGINT, signal_handler)

np.set_printoptions(edgeitems=4, linewidth=130)

np.random.seed(1) 
# TODO: with certain seeds, e.g. seed=1, the cost generates NAN

def plot_costs(costs, model_match_percents, learning_rate=0.01):
    x = np.arange(0, len(costs))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, costs, 'b*-')
    ax2.plot(x, model_match_percents, 'rx-')
    ax1.set_xlabel('Iterations (learning_rate={0})'.format(learning_rate))
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
        parameters['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.05 # the down-scaling is important
        parameters['b'+str(i)] = np.zeros((layer_dims[i], 1))
        # print(parameters['W'+str(i)].shape, parameters['b'+str(i)].shape)

    return parameters

def save_model(parameters):
    model_name = datetime.now().strftime("trained_models/%Y%m%d_%Hh%Mm") + '_model.pickle'
    pickle.dump(parameters, open(model_name, "wb"))

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
def back_prop(X, Y, parameters, caches):
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
        grads['dZ' + str(l)] = np.dot(parameters['W' + str(l+1)].T, grads['dZ' + str(l+1)]) * np.ones_like(caches['Z' + str(l)]) * (caches['Z' + str(l)] > 0.0)
        grads['dW' + str(l)] = np.dot(grads['dZ' + str(l)], caches['A' + str(l-1)].T)
        grads['db' + str(l)] = np.sum(grads['dZ' + str(l)], axis=1, keepdims=True)

    return grads

# %% [markdown] 
# Suspecting back-prop has mistakes, here is to implement the gradient check
# Design: perturb each elements in W and b by epsilon, and calculate the cost difference
#         then take the ratio as the approx gradient
def grad_check(X, Y, parameters, grads, epsilon=1e-7, light_check = True):
    m = Y.shape[1]
    grads_approx = {i:np.zeros_like(grads[i]) for i in grads} # la-la-la, dict comprehension
    for k in parameters:
        # print('Gradient check: ', k)
        if (k == 'W1') and light_check: # skip W1, because it too large for run time check
            continue
        for idx, p in np.ndenumerate(parameters[k]):
            parameters[k][idx] += epsilon
            cost_plus, tmp1, tmp2, tmp3 = forward_prop(X, Y, parameters)
            parameters[k][idx] -= 2 * epsilon
            cost_minus, tmp1, tmp2, tmp3 = forward_prop(X, Y, parameters)
            parameters[k][idx] += epsilon
            grads_approx['d'+k][idx] = (cost_plus - cost_minus) / (2 * epsilon)
            if np.abs(grads['d'+k][idx] - grads_approx['d'+k][idx]) >= epsilon:
                print('grads_approx[d'+k+']['+str(idx)+']=', grads_approx['d'+k][idx], ', grads[d'+k+']['+str(idx)+']=', grads['d'+k][idx])
                print('Gradient diff = ', grads['d'+k][idx] - grads_approx['d'+k][idx])
                print('Wrong gradient!')
            assert(np.abs(grads['d'+k][idx] - grads_approx['d'+k][idx]) <= epsilon * 100)
            
# %% [markdown]
# ### The overall model
# Here is the overall learning model with hyper-parameters
def train_model(X, Y, layer_dims, number_of_iterations = 5, learning_rate = 0.01):
    # model initialization
    global global_stop_flag
    parameters = model_init(layer_dims)

    # results
    costs = np.zeros((number_of_iterations))
    model_match_percents = np.zeros((number_of_iterations))

    for i in range(number_of_iterations):
        costs[i], caches, predicted_result, model_match_percents[i] = forward_prop(X, Y, parameters)
        
        if global_stop_flag: # If stop training earlier, then save the model
            save_model(parameters)
            sys.exit(0)
            
        # print('len(train_set_y_orig)=', len(train_set_y_orig))
        # print('train_set_y_orig.shape', train_set_y_orig.shape)
        grads = back_prop(X, Y, parameters, caches)

        if i % 100 == 0:
            dt_string = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
            print(dt_string, ' Iteration {0:5.0f}, cost={1:.6f}, prediction accuracy={2:.4f}'.format(i, costs[i], model_match_percents[i]))
            # grad_check(X, Y, parameters, grads, light_check=True)

        parameters = {k:parameters[k] - grads['d'+k] * learning_rate for k in parameters}
        # np.testing.assert_array_equal(parameters_comp['W1'], parameters['W1'])

    save_model(parameters)
    plot_costs(costs, model_match_percents, learning_rate)

    return parameters, costs

def main(train_validate_select):
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
    learning_rate = 0.0001

    if train_validate_select == 'Train':
        model, tmp1 = train_model(train_set_x, train_set_y_orig, layer_dims, 800000, learning_rate)
    elif train_validate_select == 'Validate':
        model = pickle.load(open('trained_models/20230220_23h40m_model.pickle', "rb"))
        tmp1, tmp2, tmp3, model_match_percent = forward_prop(test_set_x, test_set_y_orig, model)
        print('Test set prediction accuracy {0:.3}'.format(model_match_percent))
    else:
        print('No matched options. Options are: Train and Validate')

if __name__ == "__main__":
    main('Train')

# %% Temporary test code
# for i in range(2, 0, -1):
#     print('i=', i)
sys.exit()

