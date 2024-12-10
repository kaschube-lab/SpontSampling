import numpy as np
from heapq import nsmallest
from scipy.spatial.distance import cosine
from control_data import create_gauss

def compute_cosine_dist_matrix(x, args):
    """
    Computes the cosine distance matrix for data and the number of time frames selected.
    Args: 
        x (np.ndarray): Data array of shape (n_pixels, n_timeframes).
        args (argparse object): program arguments
    Returns:
        cosine_dist (np.ndarray): upper triangular matrix with cosine distances.
    """
    _, n_timeframes = x.shape
    cosine_dist = np.empty((n_timeframes, n_timeframes))
    for t in range(n_timeframes):
        for offset in range(args.window_size):
            try:
                cosine_dist[t, t+offset] = cosine(x[:, t], x[:, t+offset])
            except:
                continue
    return cosine_dist


def compute_avg_min_cosine_distances(x, avg_min_dist_to_preceding, avg_min_dist_to_following, args):
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
    _, n_timeframes = x.shape
    cosine_dist = compute_cosine_dist_matrix(x, args)
    for t in range(n_timeframes):
        if t >= args.window_size:
            preceding_distances = cosine_dist[t-args.window_size:t, t]
            # Get the k smallest distances
            k_smallest_preceding = nsmallest(args.k, preceding_distances)
            # Take the average of the smallest k distances
            avg_min_dist_to_preceding[t - args.window_size] = np.mean(k_smallest_preceding)

        if t <= n_timeframes - args.window_size - 1:
            # Compute distances to following patterns within the timeframe
            following_distances = cosine_dist[t, t+1: t+args.window_size+1]
            # Get the k smallest distances
            k_smallest_following = nsmallest(args.k, following_distances)
            # Take the average of the smallest k distances
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

    print('compute knn for normal data', flush=True)
    compute_avg_min_cosine_distances(x, d_results_sample['avg_min_dist_to_preceding'][j], 
                                        d_results_sample['avg_min_dist_to_following'][j], args)

    print('compute knn for Gauss', flush=True)
    compute_avg_min_cosine_distances(x_gauss, d_results_sample['avg_min_dist_to_preceding_gauss'][j], 
                                         d_results_sample['avg_min_dist_to_following_gauss'][j], args)

    print('compute knn for shuffled data', flush=True)
    for shuffle_i in range(args.n_shuffles):
        x_shuffled = np.apply_along_axis(np.random.permutation, 1, x)
        compute_avg_min_cosine_distances(x_shuffled, d_results_sample['avg_min_dist_to_preceding_random'][shuffle_i, j], 
                                        d_results_sample['avg_min_dist_to_following_random'][shuffle_i, j], args)
