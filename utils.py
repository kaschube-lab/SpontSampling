import argparse
import os

import numpy as np

def get_args():
    """
    Gets the input arguments to the program
    """
    parser = argparse.ArgumentParser(
                    prog='SpontSampling',
                    description='Computes the different metrics to assess the dynamics of spont activity')
    parser.add_argument('--functions', type=str, default='Entropy', help='Name of the function to compute. If wanting to compute several functions they need to be separated by ;')
    parser.add_argument('-d', '--data_set', type=str, default='Stringer', help='Name of the data set to use')
    parser.add_argument('--min_frames', type=int, default=5, help='Minimum number of frames to use')
    parser.add_argument('--steps', type=int, default=400, help='Number of time frames to compute the entropy for')
    parser.add_argument('--step_size', type=int, default=1, help='step size to increase the number of frames')
    parser.add_argument('--n_shuffles', type=int, default=100, help='Number of random shuffles')
    parser.add_argument('--n_inits', type=int, default=20, help='Number of starting conditions')
    parser.add_argument('--data_dir', type=str, default='./', help='directory to the data (without the data folder)')
    parser.add_argument('--save_dir', type=str, default='./', help='directory where to save results (without the results folder)')
    parser.add_argument('--window_size', type=int, default=10, help='Non-overlapping sliding window width')
    parser.add_argument('--animal_name', type=str, default='Krebs', help='Name of animal to load')
    parser.add_argument('--seed', type=int, default=0, help='initialization seed for random selection of start frames')
    parser.add_argument('--dt', type=int, default=1, help='Time steps between data points to avoid spill-over')
    parser.add_argument('--roi', type=str, default='', help='Which brain region to use')
    parser.add_argument('--EO', type=str, default='after', help='before or after eye opening')
    parser.add_argument('--condition', type=str, default='awake', help='awake or anesth animal')
    parser.add_argument('--FDiff', type=int, default=0, help='First difference (1) or not (0)')
    parser.add_argument('--dim_type', type=str, default='pr', help='Type of dimensionality measure (pr, etc.)')
    
    args = parser.parse_args()
    print(args)


def init_results_dict(n_samples, area_labels, locations, args):
    """
    Initializes a dictionary to store the results
    Parameters:
    -   n_samples (int): number of samples/trials/probes in the data
    -   area_labels (list, default None): all names of potential areas 
    -   locations (list, default None): list or np.array with the indices of the area at each location
    -   args (argparse object): arguments
    """
    d_results = {f'sample_{i}': {} for i in n_samples}
    d_results.update({'meta_data': {'data_set': args.data_set, 'n_shuffles': args.n_shuffles,
                                    'dt': args.dt, 'n_samples': n_samples}})
    if args.roi:
            d_results['meta_data']['roi'] = args.roi
    if args.data_set.lower() == 'stringer':
        d_results['meta_data']['animal_name'] = args.animal_name
        d_results['meta_data']['window_size'] = args.window_size
        d_results['meta_data']['HZ'] = 30 / args.window_size
    elif args.data_set.lower() == 'ferret':
        d_results['meta_data']['EO'] = args.EO
        d_results['meta_data']['condition'] = args.condition
        d_results['meta_data']['FDiff'] = args.FDiff
        
    if type(area_labels) is not type(None):
         d_results['meta_data']['area_labels'] = area_labels
    if type(locations) is not type(None): 
        d_results['meta_data']['locations'] = locations
    return d_results


def update_res_dict(d_results, X, function, args):
    """
    Updates the results dictionary with the keys relevant for the computed function
    Parameters:
    -   d_results (dict): results dictionary
    -   X ([np.array]): list of data matrices for each sample/trial/probe
    -   function (str): name of the function to compute
    -   args (argparse object): arguments
    Returns: 
    -   d_results (dict): results dictionary
    """
    for i, x in enumerate(X):
        if function.lower() == 'entropy':
            n_neurons, _ = x.shape
            shape_real = (args.n_inits, n_neurons, args.steps)
            shape_random = (args.n_shuffles, args.n_inits, n_neurons, args.steps)

            d_results[f'sample_{i}'].update({
                'sample_entropy': np.empty(shape_real),
                'sample_entropy_random': np.empty(shape_random),
                'approximate_entropy': np.empty(shape_real),
                'approximate_entropy_random': np.empty(shape_random),
                'fuzzy_entropy': np.empty(shape_real),
                'fuzzy_entropy_random': np.empty(shape_random),
                'weighted_permutation_entropy': np.empty(shape_real),
                'weighted_permutation_entropy_random': np.empty(shape_random),
                #'hurst_exponent': np.empty((n_neurons, args.steps)),
                #'lyapunov_exponent': np.empty((n_neurons, args.steps)),
                'fractal_dimension_katz': np.empty(shape_real),
                'fractal_dimension_katz_random': np.empty(shape_random),
                'fisher_information': np.empty(shape_real),
                'fisher_information_random': np.empty(shape_random),
                })
    
        if function.lower() == 'curvatures':
            n_neurons, n_timeframes = x.shape
            n_frames_dt = n_timeframes // args.dt
            shape_real = (args.dt, n_frames_dt - 2)
            shape_random = (args.n_shuffles, args.dt, n_frames_dt - 2)
            d_results[f'sample_{i}'].update({
                'curvatures': np.empty(shape_real),
                'curvatures_random': np.empty(shape_random)
                })
    
        if function.lower() == 'pr':
            shape_real = (args.n_inits, args.steps)
            shape_random = (args.n_shuffles, args.n_inits, args.steps)
            d_results[f'sample_{i}'].update({
                'pr': np.empty(shape_real),
                'pr_random': np.empty(shape_random)
                })
    
        if function.lower() == 'nn':
            _, n_timeframes = x.shape
            shape_real = (args.n_inits, n_timeframes, n_timeframes)
            shape_random = (args.n_shuffles, args.n_inits, n_timeframes, n_timeframes)
            d_results[f'sample_{i}'].update({
                'NN': np.empty(shape_real),
                'NN_random': np.empty(shape_random), 
                'Cosine_similarity': np.empty(shape_real),
                'Cosine_similarity_random': np.empty(shape_random)
                })


def get_save_path(args):
    """
    Generates the path and file name under which the results should be saved
    Parameters:
    -   args (argparse object): arguments
    Returns: 
    -   save_path (str): Path (incl. dir and file name) for storing the results
    """
    functions = '_'.join(args.functions.split(';'))
    res_dir = os.path.join(args.save_dir, 'results', functions)
    os.makedirs(res_dir, exist_ok=True)
    save_name = f'{functions}_{args.data_set}_{args.dt}dt'
    if args.data_set.lower() == 'stringer':
        save_name += f'_{args.animal_name}_{args.window_size}windowsize'
    elif args.data_set.lower() == 'ferret':
        fd = 'FDiff' if args.FDiff else 'Orig'
        save_name += f'{args.EO}EO_{args.condition}_{fd}'
    if args.roi:
        save_name +=  f'_{args.roi}' 
    save_name += '.npz'
    save_path = os.path.join(res_dir, save_name)
    return save_path
    