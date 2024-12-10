import numpy as np
from heapq import nsmallest
from scipy.spatial.distance import cosine
from control_data import create_gauss


def compute_min_cosine_distances(data, window_size):
    """
    Compute the minimal cosine distance of each pattern to preceding and following patterns within a timeframe.

    Args:
        data (np.ndarray): Array of shape (n_neurons, n_timeframes).
        window_size (int): The number of time frames to consider before and after.

    Returns:
        tuple: Two arrays of shape (n_timeframes,) containing the minimal cosine distances to 
               preceding and following patterns for each timepoint.
    """
    n_neurons, n_timeframes  = data.shape
    
    # Initialize arrays to store minimal distances
    min_dist_to_preceding = np.full(n_timeframes, np.inf)
    min_dist_to_following = np.full(n_timeframes, np.inf)
    
    for t in range(n_timeframes):
        # Compute distances to preceding patterns with the current timeframe
        for tf in range(max(0, t - window_size), t):
            distance = cosine(data[:, t], data[:, tf])
            min_dist_to_preceding[t] = min(min_dist_to_preceding[t], distance)
        
        # Compute distances to following patterns with the current timeframe
        for tf in range(t + 1, min(n_timeframes, t + window_size + 1)):
            distance = cosine(data[:, t], data[:, tf])
            min_dist_to_following[t] = min(min_dist_to_following[t], distance)
    
    return min_dist_to_preceding, min_dist_to_following



def compute_avg_min_cosine_distances(x, t, avg_min_dist_to_preceding, avg_min_dist_to_following, args):
    """
    Compute the average minimal cosine distance to the nearest k patterns 
    for each pattern to preceding and following patterns within a timeframe.

    Args:
        x (np.ndarray): Array of shape (n_pixels, n_timeframes).
        t (int): current time frame.
        avg_min_dist_to_preceding (np.ndarray): array to store the cosine distance to all preceeding for the current time frame.
        avg_min_dist_to_following (np.ndarray): array to store the cosine distance to all following for the current time frame.
        args (argparse object): program arguments

    Returns:
        None

    """
    if t >= args.window_size:
        preceding_distances = []
        for tf in range(max(0, t - args.window_size), t):
            distance = cosine(x[:, t], x[:, tf])
            preceding_distances.append(distance)
        
        # Take the average of the smallest k distances
        if preceding_distances:
            k_smallest_preceding = nsmallest(args.k, preceding_distances)
            avg_min_dist_to_preceding[t - args.window_size] = np.mean(k_smallest_preceding)
 
    if t <= n_timeframes - args.window_size - 1:
        # Compute distances to following patterns within the timeframe
        following_distances = []
        for tf in range(t + 1, min(n_timeframes, t + args.window_size + 1)):
            distance = cosine(x[:, t], x[:, tf])
            following_distances.append(distance)
        
        # Take the average of the smallest k distances
        if following_distances:
            k_smallest_following = nsmallest(k, following_distances)
            avg_min_dist_to_following
            avg_min_dist_to_following[t] = np.mean(k_smallest_following)



def calc_avrg_knn(x, d_results_sample, j, args):
    """
    Compute the average minimal cosine distance to the nearest k patterns 
    for each pattern to preceding and following patterns within a timeframe.

    Args:
        x (np.ndarray): Array of shape (n_pixels, n_timeframes).
        d_results_sample (dict): results dictionary for the current sample
        j (int): current iteration index
        args (argparse object): program arguments

    Returns:
        None

    """
    if args.k > args.window_size: 
        raise ValueError ('k nearest frames have to be smaller or equal to the number of time frames to consider (window_size)')

    x = x[:, j::args.dt]
    _, n_timeframes = x.shape

    if args.window_size > n_timeframes:
        raise ValueError ('The number of time frames to consider (window_size) has to be less than the number of \
            time frames in the data.')

    norm = np.linalg.norm(x, axis=0, keepdims=True)
    # Set everything to 0 that has a norm smaller than epsilon to control for the zero activity vectors.
    norm = np.where(norm < args.knn_epsilon, 0, norm)

    # Normalize the matrix along the neurons dimension
    x = x / norm

    x_gauss = create_gauss(x)


    for t in range(n_timeframes):
        # Compute distances to preceding patterns within the timeframe
        calc_avrg_knn(x, t, d_results_sample['avg_min_dist_to_preceding'][d], 
                    d_results_sample['avg_min_dist_to_following'][d], args)
        
        calc_avrg_knn(x_gauss, t, d_results_sample['avg_min_dist_to_preceding_gauss'][d], 
                    d_results_sample['avg_min_dist_to_following_gauss'][d], args)


        for shuffle_i in range(args.n_shuffles):
            X_shuffled = np.apply_along_axis(np.random.permutation, 1, x)
            calc_avrg_knn(X_shuffled, t, d_results_sample['avg_min_dist_to_preceding_random'][shuffle_i, d], 
                    d_results_sample['avg_min_dist_to_following_random'][shuffle_i, d], args)
