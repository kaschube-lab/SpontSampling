import argparse
import numpy as np
import neurokit2 as nk
import pandas as pd
import scipy.io

from data_utils import calc_windowed_spike_matrix

import warnings
warnings.filterwarnings("ignore")


def compute_entropy_measures_increasing_frames(average_spike_count, args):
    # initial_frame, steps, num_shuffles, n_inits):
    """
    Compute entropy and complexity metrics on an increasing number of time frames for each neuron.

    Parameters:
    - average_spike_count (np.ndarray): Averaged spike count matrix (neurons x frames).
    - initial_frame (int): Starting number of frames for the calculation.
    - steps (int): Number of incremental steps (i.e., compute metrics for initial_frame, initial_frame + 1, ..., initial_frame + steps).
    - num_shuffles (int): Number of times to shuffle the input data for each neuron.
    - n_inits (int): Number of starting conditions

    Returns:
    - Dict: Dictionary containing entropy and complexity metrics for each neuron and frame increment.
    """
    np.random.seed(args.seed)
    num_neurons, max_time_frames = average_spike_count.shape
    
    metrics_dict = {
        'sample_entropy': np.empty((args.n_inits, num_neurons, args.steps)),
        'sample_entropy_random': np.empty((args.num_shuffles, args.n_inits, num_neurons, args.steps)),
        'approximate_entropy': np.empty((args.n_inits, num_neurons, args.steps)),
        'approximate_entropy_random': np.empty((args.num_shuffles, args.n_inits, num_neurons, args.steps)),
        'fuzzy_entropy': np.empty((args.n_inits, num_neurons, args.steps)),
        'fuzzy_entropy_random': np.empty((args.num_shuffles, args.n_inits, num_neurons, args.steps)),
        'weighted_permutation_entropy': np.empty((args.n_inits, num_neurons, args.steps)),
        'weighted_permutation_entropy_random': np.empty((args.num_shuffles, args.n_inits, num_neurons, args.steps)),
        #'hurst_exponent': np.empty((num_neurons, args.steps)),
        #'lyapunov_exponent': np.empty((num_neurons, args.steps)),
        'fractal_dimension_katz': np.empty((args.n_inits, num_neurons, args.steps)),
        'fractal_dimension_katz_random': np.empty((args.num_shuffles, args.n_inits, num_neurons, args.steps)),
        'fisher_information': np.empty((args.n_inits, num_neurons, args.steps)),
        'fisher_information_random': np.empty((args.num_shuffles, args.n_inits, num_neurons, args.steps)),
        'initial_frame': args.initial_frame, 
        'steps': args.steps, 
        'num_shuffles': args.num_shuffles, 
        'n_inits': args.n_inits,
        'seed': args.seed,
        'shape': ('n_inits', 'n_neurons', 'steps'),
        'shape_rand': ('num_shuffles', 'n_inits', 'n_neurons', 'steps')
    }

    if args.data_set == 'Stringer':
        metrics_dict['Hz'] = 30 / args.window_size
        save_name = f'./results/Entropy_Stringer_neuropixels_{args.animal_name}_{args.seed}.npz'
    
    

    # Loop over initializations 
    for j in range(args.n_inits):
        print(j, end=': ', flush=True)
        start_frame = np.random.randint(max_time_frames - (args.steps + args.initial_frame))
        # Loop over each neuron
        for i in range(num_neurons):
            neuron_data = average_spike_count[i, :]
            print(i, end=',', flush=True)
            # Loop over the range of increasing frames
            for step in range(args.steps):
                frame_count = args.initial_frame + step
                if frame_count > max_time_frames:
                    break
                data_subset = neuron_data[start_frame:start_frame+frame_count]

                # Calculate entropy and complexity metrics for the current frame count
                
                metrics_dict['sample_entropy'][j, i, step] = nk.entropy_sample(data_subset)[0]
                metrics_dict['approximate_entropy'][j, i, step] = nk.entropy_approximate(data_subset)[0]
                metrics_dict['fuzzy_entropy'][j, i, step] = nk.entropy_fuzzy(data_subset)[0]
                metrics_dict['weighted_permutation_entropy'][j, i, step] = nk.entropy_permutation(data_subset, weighted=True)[0]
                # metrics_dict['hurst_exponent'][j, i, step] = nk.fractal_hurst(data_subset)
                # metrics_dict['lyapunov_exponent'][j, i, step] = nk.entropy_lyapunov(data_subset)
                metrics_dict['fractal_dimension_katz'][j, i, step] = nk.fractal_katz(data_subset)[0]
                metrics_dict['fisher_information'][j, i, step] = nk.fisher_information(data_subset)[0]
                
                for shuffle_i in range(args.num_shuffles):
                    shuffled_data = np.random.permutation(data_subset)
                    metrics_dict['sample_entropy_random'][shuffle_i, j, i, step] = nk.entropy_sample(shuffled_data)[0]
                    metrics_dict['approximate_entropy_random'][shuffle_i, j, i, step] = nk.entropy_approximate(shuffled_data)[0]
                    metrics_dict['fuzzy_entropy_random'][shuffle_i, j, i, step] = nk.entropy_fuzzy(shuffled_data)[0]
                    metrics_dict['weighted_permutation_entropy_random'][shuffle_i, j, i, step] = nk.entropy_permutation(shuffled_data, weighted=True)[0]
                    # metrics_dict['hurst_exponent'][shuffle_i, j, i, step] = nk.fractal_hurst(shuffled_data)
                    # metrics_dict['lyapunov_exponent'][shuffle_i, j, i, step] = nk.entropy_lyapunov(shuffled_data)
                    metrics_dict['fractal_dimension_katz_random'][shuffle_i, j, i, step] = nk.fractal_katz(shuffled_data)[0]
                    metrics_dict['fisher_information_random'][shuffle_i, j, i, step] = nk.fisher_information(shuffled_data)[0]
        print('')  
        np.savez_compressed(save_name, **metrics_dict)      

    return metrics_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-d', '--data_set', type=str, default='Stringer', help='Name of the data set to use')
    parser.add_argument('--initial_frame', type=int, default=5, help='Minimum number of frames to use')
    parser.add_argument('--steps', type=int, default=400, help='Number of time frames to compute the entropy for')
    parser.add_argument('--num_shuffles', type=int, default=100, help='Number of random shuffles')
    parser.add_argument('--n_inits', type=int, default=20, help='Number of starting conditions')
    parser.add_argument('--window_size', type=int, default=10, help='Non-overlapping sliding window width')
    parser.add_argument('--animal_name', type=str, default='Krebs', help='Name of animal to load')
    parser.add_argument('--seed', type=int, default=0, help='initialization seed for random selection of start frames')
    
    args = parser.parse_args()
    print(args)

    if args.data_set == 'Stringer':
        
        f = scipy.io.loadmat(f'./data/StringerNeuropixels/{args.animal_name}withFaces_KS2.mat')
        data_matrix = calc_windowed_spike_matrix(f['stall'], args.window_size)
        area_labels = [x[0] for x in f['areaLabels'][0]]
        print('data_matrix.shape', data_matrix.shape)

    entropy_results = compute_entropy_measures_increasing_frames(data_matrix, args)
    #args.initial_frame, args.steps, 
    #                                                            args.num_suffles, args.n_inits, Hz)

    if args.data_set == 'Stringer':
        # entropy_results['Hz'] = 30 / args.window_size
        np.savez_compressed(f'./results/Entropy_Stringer_neuropixels_{args.animal_name}_{args.seed}.npz', **entropy_results)