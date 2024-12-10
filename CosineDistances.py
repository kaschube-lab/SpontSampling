import numpy as np


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


def compute_avg_min_cosine_distances(X, timeframe, k):
    """
    Compute the average minimal cosine distance to the nearest k patterns 
    for each pattern to preceding and following patterns within a timeframe.

    Args:
        X (np.ndarray): Array of shape (n_pixels, n_timeframes).
        max_tfs (int): The number of timeframes to consider before and after.
        k (int): Number of nearest patterns to average.

    Returns:
        tuple: Two arrays of shape (n_timeframes,) containing the average minimal cosine 
               distances to preceding and following patterns for each timeframe.
    """
    _, n_timeframes = data.shape
    
    # Initialize arrays to store average minimal distances
    avg_min_dist_to_preceding = np.full(n_timeframes, np.inf)
    avg_min_dist_to_following = np.full(n_timeframes, np.inf)
    
    for t in range(n_timeframes):
        # Compute distances to preceding patterns within the timeframe
        preceding_distances = []
        for tf in range(max(0, t - max_tfs), t):
            distance = cosine(data[:, t], data[:, tf])
            preceding_distances.append(distance)
        
        # Take the average of the smallest k distances
        if preceding_distances:
            k_smallest_preceding = nsmallest(k, preceding_distances)
            avg_min_dist_to_preceding[t] = np.mean(k_smallest_preceding)
        
        # Compute distances to following patterns within the timeframe
        following_distances = []
        for tf in range(t + 1, min(n_timeframes, t + max_tfs + 1)):
            distance = cosine(data[:, t], data[:, tf])
            following_distances.append(distance)
        
        # Take the average of the smallest k distances
        if following_distances:
            k_smallest_following = nsmallest(k, following_distances)
            avg_min_dist_to_following[t] = np.mean(k_smallest_following)
    
    return avg_min_dist_to_preceding, avg_min_dist_to_following