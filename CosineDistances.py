import numpy as np
from heapq import nsmallest
from scipy.spatial.distance import cosine


def compute_min_cosine_distances(data, max_tfs):
    """
    Compute the minimal cosine distance of each pattern to preceding and following patterns within a timeframe.

    Args:
        data (np.ndarray): Array of shape (n_neurons, n_timeframes).
        max_tfs (int): The number of time frames to consider before and after.

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
        for tf in range(max(0, t - max_tfs), t):
            distance = cosine(data[:, t], data[:, tf])
            min_dist_to_preceding[t] = min(min_dist_to_preceding[t], distance)
        
        # Compute distances to following patterns with the current timeframe
        for tf in range(t + 1, min(n_timeframes, t + max_tfs + 1)):
            distance = cosine(data[:, t], data[:, tf])
            min_dist_to_following[t] = min(min_dist_to_following[t], distance)
    
    return min_dist_to_preceding, min_dist_to_following


def compute_avg_min_cosine_distances(x, d_results_sample, args):
    """
    Compute the average minimal cosine distance to the nearest k patterns 
    for each pattern to preceding and following patterns within a timeframe.

    Args:
        x (np.ndarray): Array of shape (n_pixels, n_timeframes).
        d_results_sample (dict): results dictionary for the current sample
        args (argparse object): program arguments

    Returns:
        avg_min_dist_to_preceding (np.ndarray): average min cosine distance to all preceeding frames for each time frame. 
                                                Shape: (n_timeframes - max_tfs, )
        avg_min_dist_to_following (np.ndarray): average min cosine distance to all following frames for each time frame. 
                                                Shape: (n_timeframes - max_tfs, )

    """
    if args.k > args.max_tfs: 
        raise ValueError ('k nearest frames have to be smaller or equal to the number of time frames to consider (max_tfs)')

    
    norm = np.linalg.norm(x, axis=0, keepdims=True)
    norm = np.where(norm < args.knn_epsilon, 0, norm)

    # Normalize the matrix along the neurons dimension
    x = x / norm


    for d in range(args.dt):
        x_subset = x[:, d::args.dt]
        _, n_timeframes = x_subset.shape

        if args.max_tfs > n_timeframes:
            raise ValueError ('The number of time frames to consider (max_tfs) has to be less than the number of \
                time frames in the data.')

    
        for t in range(n_timeframes):
            # Compute distances to preceding patterns within the timeframe
            if t >= max_tfs:
                preceding_distances = []
                for tf in range(max(0, t - max_tfs), t):
                    distance = cosine(x_subset[:, t], x_subset[:, tf])
                    preceding_distances.append(distance)
                
                # Take the average of the smallest k distances
                if preceding_distances:
                    k_smallest_preceding = nsmallest(k, preceding_distances)
                    d_results_sample['avg_min_dist_to_preceding'][d, t - max_tfs] = np.mean(k_smallest_preceding)
            
            if t <= n_timeframes - max_tfs - 1:
                # Compute distances to following patterns within the timeframe
                following_distances = []
                for tf in range(t + 1, min(n_timeframes, t + max_tfs + 1)):
                    distance = cosine(x_subset[:, t], x_subset[:, tf])
                    following_distances.append(distance)
                
                # Take the average of the smallest k distances
                if following_distances:
                    k_smallest_following = nsmallest(k, following_distances)
                    d_results_sample['avg_min_dist_to_following'][d, t] = np.mean(k_smallest_following)
    