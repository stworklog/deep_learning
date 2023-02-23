# Analyze the model parameters from different trainings
import numpy as np
import pickle

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

if __name__ == "__main__":
    np.set_printoptions(edgeitems=4, linewidth=130)
    main()
