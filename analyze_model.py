# Analyze the model parameters from different trainings
import numpy as np
import pickle
import seaborn
import matplotlib.pyplot as plt

# create interactive data visualization using bokeh to visualize norms of weights and biases
def plot_norms(norms, norm_type='fro'):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].set_title('Norm of model_1 ini')
    axs[0, 0].bar(norms[norm_type]['m1_ini'].keys(), norms[norm_type]['m1_ini'].values())
    axs[0, 1].set_title('Norm of model_2 ini')
    axs[0, 1].bar(norms[norm_type]['m2_ini'].keys(), norms[norm_type]['m2_ini'].values())
    axs[1, 0].set_title('Norm of trained model_1')
    axs[1, 0].bar(norms[norm_type]['m1'].keys(), norms[norm_type]['m1'].values())
    axs[1, 1].set_title('Norm of trained model_2')
    axs[1, 1].bar(norms[norm_type]['m2'].keys(), norms[norm_type]['m2'].values())
    plt.show()

def plot_norms_diff(norms, norm_type='fro'):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].set_title('Norm of m1_ini_diff_m2_ini')
    axs[0, 0].bar(norms[norm_type]['m1_ini_diff_m2_ini'].keys(), norms[norm_type]['m1_ini_diff_m2_ini'].values())
    axs[0, 1].set_title('Norm of m1_diff_m1_ini')
    axs[0, 1].bar(norms[norm_type]['m1_diff_m1_ini'].keys(), norms[norm_type]['m1_diff_m1_ini'].values())
    axs[1, 0].set_title('Norm of m2_diff_m2_ini')
    axs[1, 0].bar(norms[norm_type]['m2_diff_m2_ini'].keys(), norms[norm_type]['m2_diff_m2_ini'].values())
    axs[1, 1].set_title('Norm of m1_diff_m2')
    axs[1, 1].bar(norms[norm_type]['m1_diff_m2'].keys(), norms[norm_type]['m1_diff_m2'].values())
    plt.show()


def calc_norms(model_1, model_2):
    p = 'parameters'
    p_ini = p + '_ini'
    norm_fro = {}
    norm_fro['m1_ini'] = {k:np.linalg.norm(model_1[p_ini][k], ord='fro') for k in model_1[p_ini]}
    norm_fro['m2_ini'] = {k:np.linalg.norm(model_2[p_ini][k], ord='fro') for k in model_2[p_ini]}
    norm_fro['m1'] = {k:np.linalg.norm(model_1[p][k], ord='fro') for k in model_1[p]}
    norm_fro['m2'] = {k:np.linalg.norm(model_2[p][k], ord='fro') for k in model_2[p]}
    norm_fro['m1_ini_diff_m2_ini'] = {k:np.linalg.norm(model_1[p_ini][k] - model_2[p_ini][k], ord='fro') for k in model_1[p_ini]}
    norm_fro['m1_diff_m1_ini'] = {k:np.linalg.norm(model_1[p][k] - model_1[p_ini][k], ord='fro') for k in model_1[p]}
    norm_fro['m2_diff_m2_ini'] = {k:np.linalg.norm(model_2[p][k] - model_2[p_ini][k], ord='fro') for k in model_2[p]}
    norm_fro['m1_diff_m2'] = {k:np.linalg.norm(model_1[p][k] - model_2[p][k], ord='fro') for k in model_1[p]}

    norm_2 = {}
    norm_2['m1_ini'] = {k:np.linalg.norm(model_1[p_ini][k], ord=2) for k in model_1[p_ini]}
    norm_2['m2_ini'] = {k:np.linalg.norm(model_2[p_ini][k], ord=2) for k in model_2[p_ini]}
    norm_2['m1'] = {k:np.linalg.norm(model_1[p][k], ord=2) for k in model_1[p]}
    norm_2['m2'] = {k:np.linalg.norm(model_2[p][k], ord=2) for k in model_2[p]}
    norm_2['m1_ini_diff_m2_ini'] = {k:np.linalg.norm(model_1[p_ini][k] - model_2[p_ini][k], ord=2) for k in model_1[p_ini]}
    norm_2['m1_diff_m1_ini'] = {k:np.linalg.norm(model_1[p][k] - model_1[p_ini][k], ord=2) for k in model_1[p]}
    norm_2['m2_diff_m2_ini'] = {k:np.linalg.norm(model_2[p][k] - model_2[p_ini][k], ord=2) for k in model_2[p]}
    norm_2['m1_diff_m2'] = {k:np.linalg.norm(model_1[p][k] - model_2[p][k], ord=2) for k in model_1[p]}

    norms = {}
    norms['fro'] = norm_fro
    norms['2'] = norm_2
    return norms
    
def main():
    model_1 = pickle.load(open('trained_models/20230221_14h18m_model.pickle', "rb"))
    model_2 = pickle.load(open('trained_models/20230221_22h26m_model.pickle', "rb"))
    # if model_1['layer_dims'] != model_2['layer_dims']:
    #     print('Comparing weights and biases from models', 
    #         'with different structures may not make sense')

    norms = calc_norms(model_1, model_2)
    plot_norms(norms, norm_type='fro')
    plot_norms_diff(norms, norm_type='fro')

if __name__ == "__main__":
    np.set_printoptions(edgeitems=4, linewidth=130)
    main()
