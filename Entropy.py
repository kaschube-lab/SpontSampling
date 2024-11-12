import os
import argparse
import numpy as np
import neurokit2 as nk
import pandas as pd
import scipy.io

from data_utils import load_data

import warnings
warnings.filterwarnings("ignore")


def compute_entropy_measures_increasing_frames(X, d_results, save_name, sample_i, args):
    # initial_frame, steps, n_shuffles, n_inits):
    """
    Compute entropy and complexity metrics on an increasing number of time frames for each neuron.

    Parameters:
    - X (np.ndarray): data matrix (neurons x frames).
    - d_results (dict): results dictionalry
    - save_name (str): where to save the data (needed for intermediate saves)
    - sample_i (int): current sample or probe to analyze
    - args (argparse object): program arguments


    Returns:
    - d (dict): Dictionary containing entropy and complexity metrics for each neuron and frame increment.
    """
    np.random.seed(args.seed)
    n_neurons, n_timeframes = X.shape
    n_frames_dt = n_timeframes // args.dt
    X = X[:, :int(n_frames_dt*args.dt)]
    

    # Add initializations for the entropy computation to the results dictionary
    d_results[f'sample_{sample_i}'] = {
        'sample_entropy': np.empty((args.n_inits, n_neurons, args.steps)),
        'sample_entropy_random': np.empty((args.n_shuffles, args.n_inits, n_neurons, args.steps)),
        'approximate_entropy': np.empty((args.n_inits, n_neurons, args.steps)),
        'approximate_entropy_random': np.empty((args.n_shuffles, args.n_inits, n_neurons, args.steps)),
        'fuzzy_entropy': np.empty((args.n_inits, n_neurons, args.steps)),
        'fuzzy_entropy_random': np.empty((args.n_shuffles, args.n_inits, n_neurons, args.steps)),
        'weighted_permutation_entropy': np.empty((args.n_inits, n_neurons, args.steps)),
        'weighted_permutation_entropy_random': np.empty((args.n_shuffles, args.n_inits, n_neurons, args.steps)),
        #'hurst_exponent': np.empty((n_neurons, args.steps)),
        #'lyapunov_exponent': np.empty((n_neurons, args.steps)),
        'fractal_dimension_katz': np.empty((args.n_inits, n_neurons, args.steps)),
        'fractal_dimension_katz_random': np.empty((args.n_shuffles, args.n_inits, n_neurons, args.steps)),
        'fisher_information': np.empty((args.n_inits, n_neurons, args.steps)),
        'fisher_information_random': np.empty((args.n_shuffles, args.n_inits, n_neurons, args.steps)),
        }


    # Loop over initializations 
    for j in range(args.n_inits):
        print(j, end=': ', flush=True)
        start_frame = np.random.randint(n_frames_dt - (args.steps + args.min_frames))
        X_subset = X[:, start_frame::args.dt]
        n_timeframes = X_subset.shape[-1]
        # Loop over each neuron
        for i in range(n_neurons):
            print(i, end=',', flush=True)
            # Loop over the range of increasing frames
            for step in range(0, args.steps, args.step_size):
                frame_count = args.min_frames + step
                if frame_count > n_timeframes:
                    break
                x = X_subset[i, start_frame:start_frame+frame_count]

                # Calculate entropy and complexity metrics for the current frame count
                d_results[f'sample_{sample_i}']['sample_entropy'][j, i, step] = nk.entropy_sample(x)[0]
                d_results[f'sample_{sample_i}']['approximate_entropy'][j, i, step] = nk.entropy_approximate(x)[0]
                d_results[f'sample_{sample_i}']['fuzzy_entropy'][j, i, step] = nk.entropy_fuzzy(x)[0]
                d_results[f'sample_{sample_i}']['weighted_permutation_entropy'][j, i, step] = nk.entropy_permutation(x, weighted=True)[0]
                # d_results[f'sample_{sample_i}']['hurst_exponent'][j, i, step] = nk.fractal_hurst(x)
                # d_results[f'sample_{sample_i}']['lyapunov_exponent'][j, i, step] = nk.entropy_lyapunov(x)
                d_results[f'sample_{sample_i}']['fractal_dimension_katz'][j, i, step] = nk.fractal_katz(x)[0]
                d_results[f'sample_{sample_i}']['fisher_information'][j, i, step] = nk.fisher_information(x)[0]
                
                for shuffle_i in range(args.n_shuffles):
                    shuffled_data = np.random.permutation(x)
                    d_results[f'sample_{sample_i}']['sample_entropy_random'][shuffle_i, j, i, step] = nk.entropy_sample(shuffled_data)[0]
                    d_results[f'sample_{sample_i}']['approximate_entropy_random'][shuffle_i, j, i, step] = nk.entropy_approximate(shuffled_data)[0]
                    d_results[f'sample_{sample_i}']['fuzzy_entropy_random'][shuffle_i, j, i, step] = nk.entropy_fuzzy(shuffled_data)[0]
                    d_results[f'sample_{sample_i}']['weighted_permutation_entropy_random'][shuffle_i, j, i, step] = nk.entropy_permutation(shuffled_data, weighted=True)[0]
                    # d_results[f'sample_{sample_i}']['hurst_exponent'][shuffle_i, j, i, step] = nk.fractal_hurst(shuffled_data)
                    # d_results[f'sample_{sample_i}']['lyapunov_exponent'][shuffle_i, j, i, step] = nk.entropy_lyapunov(shuffled_data)
                    d_results[f'sample_{sample_i}']['fractal_dimension_katz_random'][shuffle_i, j, i, step] = nk.fractal_katz(shuffled_data)[0]
                    d_results[f'sample_{sample_i}']['fisher_information_random'][shuffle_i, j, i, step] = nk.fisher_information(shuffled_data)[0]
        print('')  
        # Intermediate save, as the computation takes long and in case of a program failure and to analyze intermediate results
        np.savez_compressed(save_name, **d_results)      

    return d_results


def reformant_entropy_stringer():
    """
    Because stringer data took long to compute the entropy, I split it in batches of 5. 
    This code adds it together and also saves a single file for each metric. 
    """
    metrics = ['sample_entropy', 'approximate_entropy', 'fuzzy_entropy', 'weighted_permutation_entropy',
           'fractal_dimension_katz', 'fisher_information']

    mice = ['Krebs',  'Robbins', 'Waksman']

    for mouse in mice:
        print(mouse, end=',')
        f = scipy.io.loadmat(f'./data/StringerNeuropixels/{mouse}withFaces_KS2.mat')
        area_labels = [x[0] for x in f['areaLabels'][0]]
        locations = f['brainLoc']
        ds = []
        for i in range(4):
            ds.append(np.load(f'./results/Entropy/Entropy_Stringer_neuropixels_{mouse}_{i}.npz'))
        for metric in metrics:
            print(metric, end='. ')
            entropy_results, entropy_random = [], []
            for d in ds:
                entropy_results.append(d[metric])
                entropy_random.append(d[f'{metric}_random'])
            entropy_results = np.concatenate(entropy_results, axis=0)
            entropy_random = np.concatenate(entropy_random, axis=1)
            d_dict = dict(d)
            d_new = {metric: entropy_results, 
                    f'{metric}_random': entropy_random, 
                    'meta_data': {
                        'area_labels': area_labels,
                        'locations': locations,
                        'initial_frame': d_dict['initial_frame'],
                        'steps': d_dict['steps'],
                        'n_shuffles': d_dict['num_shuffles'],
                        'n_inits': d_dict['n_inits'],
                        'seed': d_dict['seed'],
                        'shape': d_dict['shape'],
                        'shape_rand': d_dict['shape_rand'],
                        'Hz': d_dict['Hz']                
                    }}
            np.savez_compressed(f'./results/Entropy/Entropy_Stringer_neuropixels_{metric}_{mouse}.npz', **d_new)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='Entropy',
                    description='Computes the change in entropy in the data with an increasing number of frames \
                    across different random starting points')
    parser.add_argument('-d', '--data_set', type=str, default='Stringer', help='Name of the data set to use')
    parser.add_argument('--min_frames', type=int, default=5, help='Minimum number of frames to use')
    parser.add_argument('--steps', type=int, default=400, help='Number of time frames to compute the entropy for')
    parser.add_argument('--step_size', type=int, default=1, help='step size to increase the number of frames')
    parser.add_argument('--n_shuffles', type=int, default=100, help='Number of random shuffles')
    parser.add_argument('--n_inits', type=int, default=20, help='Number of starting conditions')
    parser.add_argument('--data_dir', type=str, default='./', help='directory to the data (without the data folder)')
    parser.add_argument('--window_size', type=int, default=10, help='Non-overlapping sliding window width')
    parser.add_argument('--animal_name', type=str, default='Krebs', help='Name of animal to load')
    parser.add_argument('--seed', type=int, default=0, help='initialization seed for random selection of start frames')
    parser.add_argument('--dt', type=int, default=1, help='Time steps between data points to avoid spill-over')
    parser.add_argument('--roi', type=str, default='', help='Which brain region to use')
    parser.add_argument('--EO', type=str, default='after', help='before or after eye opening')
    parser.add_argument('--condition', type=str, default='awake', help='awake or anesth animal')
    parser.add_argument('--FDiff', type=int, default=0, help='First difference (1) or not (0)')
    
    args = parser.parse_args()
    print(args)
    X, d, save_name = load_data(args, 'Entropy')

    d['meta_data'].update({'min_frames': args.min_frames, 
                            'steps': args.steps, 
                            'n_shuffles': args.n_shuffles, 
                            'n_inits': args.n_inits,
                            'seed': args.seed,
                            'shape': ('n_inits', 'n_neurons', 'steps'),
                            'shape_rand': ('n_shuffles', 'n_inits', 'n_neurons', 'steps')})

    
    for sample_i, x in enumerate(X):
        if sample_i == 0:
            print('x.shape', x.shape)
        if sample_i % 100 == 0:
            print(sample_i, end=',')

        d = compute_entropy_measures_increasing_frames(x, d, save_name, sample_i, args)

    np.savez_compressed(save_name, **d)
