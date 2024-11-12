import numpy as np
from sklearn.decomposition import PCA


def calc_pr(X):
    """
    Method to calculate the participation ratio. 
    X (np array): data array with shape (number_pixels, number_time_frames)
    """
    pca = PCA()
    # The sklearn PCA fit() function takes input in the format (n_samples, n_features). As the time 
    # frames are the features, the matrix has to be transposed. 
    pca.fit(X.T) 
    eigenvalues = pca.explained_variance_
    pr = np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2)
    return pr


def calc_dimensionality_increasing_frames(X, d_results, save_name, sample_i, args):
    # initial_frame, steps, n_shuffles, n_inits):
    """
    Computes dimensionality on an increasing number of time frames across neurons/pixels/rois.

    Parameters:
    - X (np.ndarray): data matrix (neurons x frames).
    - d_results (dict): results dictionalry
    - save_name (str): where to save the data (needed for intermediate saves)
    - sample_i (int): current sample or probe to analyze
    - args (argparse object): program arguments


    Returns:
    - d (dict): Dictionary containing dimensionality with increasing number of time frames.
    """
    np.random.seed(args.seed)
    n_neurons, n_timeframes = X.shape
    n_frames_dt = n_timeframes // args.dt

    X = X[:, :int(n_frames_dt*args.dt)]

    # Add initializations for the dimensionality computation to the results dictionary
    d_results[f'sample_{sample_i}'] = {
        'dimensionality': np.empty((args.n_inits, n_neurons, args.steps)),
        'dimensionality_random': np.empty((args.n_shuffles, args.n_inits, n_neurons, args.steps))
        }


    # Loop over initializations 
    for j in range(args.n_inits):
        print(j, end=': ', flush=True)
        
        start_frame = np.random.randint(n_frames_dt - (args.steps + args.min_frames))
        X_subset = X[:, start_frame::args.dt]
        n_timeframes = X_subset.shape[-1]
        # Loop over each neuron
        for step in range(0, args.steps, args.step_size):
            frame_count = args.min_frames + step
            if frame_count > n_timeframes:
                break
            x = X_subset[:, start_frame:start_frame+frame_count]

            # Calculate dimensionality for the current frame count
            dim = calc_pr(x)
            d_results[f'sample_{sample_i}']['dimensionality'][j, i, step] = dim
            
        for shuffle_i in range(args.n_shuffles):
            X_shuffled = np.apply_along_axis(np.random.permutation, 1, X_subset)
            for step in range(0, args.steps, args.step_size):
                frame_count = args.min_frames + step
                if frame_count > n_timeframes:
                    break
                x = X_shuffled[:, start_frame:start_frame+frame_count]

                dim = calc_pr(x)
                d_results[f'sample_{sample_i}']['dimensionality_random'][shuffle_i, j, i, step] = dim
            
        print('')  
        # Intermediate save, as the computation takes long and in case of a program failure and to analyze intermediate results
        np.savez_compressed(save_name, **d_results)      

    return d_results

if __name__ == '__main__':

    
    args = get_args()

    X, d, save_name = load_data(args, 'Dimensionality')

    d['meta_data'].update({'min_frames': args.min_frames, 
                            'steps': args.steps, 
                            'n_shuffles': args.n_shuffles, 
                            'n_inits': args.n_inits,
                            'seed': args.seed,
                            'shape': ('n_inits', 'n_neurons', 'steps'),
                            'shape_rand': ('n_shuffles', 'n_inits', 'n_neurons', 'steps'),
                            'dt': args.dt})

    
    for sample_i, x in enumerate(X):
        if sample_i == 0:
            print('x.shape', x.shape)
        if sample_i % 100 == 0:
            print(sample_i, end=',')

        d = calc_dimensionality_increasing_frames(X, d_results, save_name, sample_i, args)

    np.savez_compressed(save_name, **d)
