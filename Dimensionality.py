import numpy as np
from sklearn.decomposition import PCA


def calc_pr(X):
    """
    Method to calculate the participation ratio. 
    Parametres:
        X (np.array): data array with shape (number_pixels, number_time_frames)

    Returns:
    - pr (float): computed participation ratio
    """
    pca = PCA()
    # The sklearn PCA fit() function takes input in the format (n_samples, n_features). As the time 
    # frames are the features, the matrix has to be transposed. 
    pca.fit(X.T) 
    eigenvalues = pca.explained_variance_
    pr = np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2)
    return pr


def calc_dimensionality_increasing_frames(X, d_results, j, args):
    # initial_frame, steps, n_shuffles, n_inits):
    """
    Computes dimensionality on an increasing number of time frames across neurons/pixels/rois.

    Parameters:
    - X (np.ndarray): data matrix (neurons x frames).
    - d_results (dict): results dictionalry
    - j (int): current iteration index
    - args (argparse object): program arguments
    """
    
        
    start_frame = np.random.randint(X.shape[-1] // args.dt - (args.steps + args.min_frames))
    X_subset = X[:, start_frame::args.dt]

    for step in range(0, args.steps, args.step_size):
        frame_count = args.min_frames + step
        x = X_subset[:, start_frame:start_frame+frame_count]

        # Calculate dimensionality for the current frame count
        if args.dim_type == 'pr':
            dim = calc_pr(x)
        d_results_sample[f'{args.dim_type}'][j, step] = dim
        
    for shuffle_i in range(args.n_shuffles):
        X_shuffled = np.apply_along_axis(np.random.permutation, 1, X_subset)
        for step in range(0, args.steps, args.step_size):
            frame_count = args.min_frames + step
            x = X_shuffled[:, start_frame:start_frame+frame_count]
            if args.dim_type == 'pr': 
                dim = calc_pr(x)
            d_results_sample[f'{args.dim_type}_random'][shuffle_i, j, step] = dim
