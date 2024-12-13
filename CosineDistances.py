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
    for t in range(n_timeframes - args.window_size):
        for offset in range(args.window_size):
            try:
                cosine_dist[t, t+offset] = cosine(x[:, t], x[:, t+offset])
            except IndexError:
               continue
    return cosine_dist


def compute_min_cosine_distances(x, min_dist_to_pre, min_dist_to_post, pre_tfs, post_tfs, args):
    """
    Compute the average minimal cosine distance to the nearest k patterns 
    for each pattern to preceding and following patterns within a timeframe.

    Args:
        x (np.ndarray): Array of shape (n_pixels, n_timeframes).
        t (int): current time frame.
        min_dist_to_pre (np.ndarray): array to store the cosine distance to all preceeding for the current time frame.
        min_dist_to_post (np.ndarray): array to store the cosine distance to all following for the current time frame.
        pre_tfs (np.ndarray): array to store the distance in indices to all preceeding for the current time frame.
        post_tfs (np.ndarray): array to store the distance in indices to all following for the current time frame.
        args (argparse object): program arguments

    Returns:
        None

    """
    _, n_timeframes = x.shape
    cosine_dist = compute_cosine_dist_matrix(x, args)
    for t in range(n_timeframes):
        if t >= args.window_size:
            preceding_distances = cosine_dist[t-args.window_size:t, t]
            # Sort distances
            sorted_distances = np.argsort(preceding_distances)
            # save distance indices
            pre_tfs[t - args.window_size] = sorted_distances[:args.k]
            # Get the k smallest distances
            k_smallest_pre = preceding_distances[sorted_distances[:args.k]] #nsmallest(args.k, preceding_distances)
            # Take the average of the smallest k distances
            min_dist_to_pre[t - args.window_size] = k_smallest_pre


        if t <= n_timeframes - args.window_size - 1:
            # Compute distances to following patterns within the timeframe
            post_distances = cosine_dist[t, t+1: t+args.window_size+1]
            # Sort distances
            sorted_distances = np.argsort(post_distances)
            # save distance indices
            post_tfs[t] = sorted_distances[:args.k]
            # Get the k smallest distances
            k_smallest_post = post_distances[sorted_distances[:args.k]] # nsmallest(args.k, post_distances)
            # Take the average of the smallest k distances
            min_dist_to_post[t] = k_smallest_post



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
    x = np.where(norm <= args.knn_epsilon, np.nan, x)
    
    

    # Normalize the matrix along the neurons dimension
    # x = x / norm
    # compute_min_cosine_distances(x, min_dist_to_pre, min_dist_to_post, pre_tfs, post_tfs, args)
    print('compute knn for normal data', flush=True)
    compute_min_cosine_distances(x, d_results_sample['min_dist_to_pre'][j], d_results_sample['min_dist_to_post'][j],
                                 d_results_sample['dist_index_pre'][j], d_results_sample['dist_index_post'][j], 
                                 args)

    print('compute knn for Gauss', flush=True)
    x_gauss = create_gauss(x)
    compute_min_cosine_distances(x_gauss, d_results_sample['min_dist_to_pre_gauss'][j], 
                                         d_results_sample['min_dist_to_post_gauss'][j], 
                                         d_results_sample['dist_index_pre_gauss'][j], 
                                         d_results_sample['dist_index_post_gauss'][j],
                                         args)

    print('compute knn for shuffled data', flush=True)
    for shuffle_i in range(args.n_shuffles):
        x_shuffled = x[:, np.random.permutation(x.shape[-1])]
        compute_min_cosine_distances(x_shuffled, d_results_sample['min_dist_to_pre_random'][shuffle_i, j], 
                                        d_results_sample['min_dist_to_post_random'][shuffle_i, j], 
                                        d_results_sample['dist_index_pre_random'][shuffle_i, j], 
                                        d_results_sample['dist_index_post_random'][shuffle_i, j], args)
