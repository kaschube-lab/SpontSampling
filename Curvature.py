import numpy as np
import math


def calc_curvature(X, degrees=False):
    """
    Compute the average curvature for a matrix X (n_neurons x n_timeframes).

    Parameters:
    - X (np.ndarray): The data matrix with shape (n_neurons, n_timeframes).
    - degrees (bool): Compute curvature in degrees or radian. 
    
    Returns:
    - C, float average curvature
    - curvatures, list, curvatures at each time point
    """
    n_neurons, n_timeframes = X.shape
    # Compute the vectors v_k = x_{k+1} - x_k for each time frame
    v = X[:, 1:] - X[:, :-1]  # Shape: (n_neurons, n_timeframes - 1)
    
    # Initialize array to hold curvatures
    curvatures = []

    # Calculate curvature c_k for each consecutive pair of vectors
    for k in range(n_timeframes - 2):
        v_k = v[:, k]
        v_k_plus_1 = v[:, k + 1]

        # Compute the dot product and norms
        dot_product = np.dot(v_k, v_k_plus_1)
        norm_v_k = np.linalg.norm(v_k)
        norm_v_k_plus_1 = np.linalg.norm(v_k_plus_1)

        # Compute curvature c_k, handling division by zero in case of zero norms
        
        if norm_v_k > 0 and norm_v_k_plus_1 > 0:
            c_k = np.arccos(dot_product / (norm_v_k * norm_v_k_plus_1))
            if degrees:
                c_k = math.degrees(c_k)
            # print(c_k, end=',')
            curvatures.append(c_k)
        else:
            curvatures.append(0)  # Assign 0 if one of the vectors has zero magnitude

    # Compute the average curvature C
    return curvatures


def calc_curvature_incrasing(X, initial_frame, steps, num_shuffles=0, n_inits=0, dt=0):
    """
    Compute the average curvature for a matrix X (n_neurons x n_timeframes).

    Parameters:
    - X (np.ndarray): The data matrix with shape (n_neurons, n_timeframes).
    - initial_frame (int): Starting number of frames for calculating curvature.
    - steps (int): Number of steps to increase the window size.
    - num_shuffles (int): Number of shuffled iterations. Default is 0 (no shuffling).
    - n_inits (int): Number of different starting conditions to compute curvature.
    - dt (int): indicates how many frames to skip to avoid spillover effects. 
    
    Returns:
    - dict: Dictionary containing:
      - 'avrg_curvature_real': Array of shape (steps,) with curvature for real data.
      - 'avrg_curvature_shuffled': Array of shape (steps, num_shuffles) with curvature for shuffled data.
      - 'curvature_real': Array of shape (steps, steps) with curvature for real data.
      - 'curvature_shuffled': Array of shape (num_shuffles, steps, steps) with curvature for shuffled data.
    """

    np.random.seed(0)
    
    n_neurons, n_timeframes = X.shape
    avrg_curvatures_real = np.empty((n_inits, steps))
    avrg_curvatures_shuffled = np.empty((num_shuffles, n_inits, steps)) if num_shuffles > 0 else None
    curvatures_real = np.empty((n_inits, steps))
    curvatures_shuffled = np.empty((num_shuffles, n_inits, steps)) if num_shuffles > 0 else None


    for i in range(n_inits):
        print(i, end=': ', flush=True)
        start_frame = np.random.randint(n_timeframes - steps - 2)
        X_subset = X[:, start_frame:start_frame+steps + 2] # +2 as the curvature needs 3 frames to be computed
        curvatures = calc_curvature(X_subset)
        curvatures_real[i] = curvatures
        for step in range(steps):
            frame_count = initial_frame + step
            if frame_count > n_timeframes:
                break
            avrg_curvatures_real[i, step] = np.mean(curvatures[:frame_count])
            #X_subset = X[:, start_frame:start_frame+frame_count]
            #curv = calc_curvature(X_subset)
            #C = np.mean(curvatures) if curvatures else 0
            # avrg_curvatures_real[i, step], curvatures_real[i, step, :frame_count-2] = calc_curvature(X_subset)

        if num_shuffles <= 0:
            continue

        for shuffle in range(num_shuffles):
            X_shuffled = np.apply_along_axis(np.random.permutation, 1, X_subset)
            curvatures = calc_curvature(X_shuffled)
            curvatures_shuffled[shuffle, i] = curvatures
            for step in range(steps):
                frame_count = initial_frame + step
                if frame_count > n_timeframes:
                    break
                avrg_curvatures_shuffled[shuffle, i, step] = np.nanmean(curvatures[:frame_count])
            #C, curv = calc_curvature(X_shuffled)
            #avrg_curvatures_shuffled[shuffle, i, step] = C
            #curvatures_shuffled[shuffle, i, step, :frame_count-2] = curv
    d = {'avrg_curvatures_real': avrg_curvatures_real,
         'avrg_curvatures_shuffled': avrg_curvatures_shuffled, 
         'curvature_real': curvatures_real, 
         'curvature_shuffled': curvatures_shuffled}
    return d


def calc_curvature_full(x, d_results_sample, args):
    n_neurons, n_timeframes = X.shape
    n_frames_dt = n_timeframes // args.dt

    x = x[:, :int(n_frames_dt*args.dt)]

    for d in range(args.dt):
        x_subset = x[:, d::args.dt]
        # print('X_subset.shape', X_subset.shape, 'n_frames_dt - 2', n_frames_dt - 2)
        d_results_sample['curvatures'][d] = calc_curvature(x_subset, degrees=True)

        for shuffle in range(args.n_shuffle):
            X_shuffled = np.apply_along_axis(np.random.permutation, 1, x_subset)
            d_results_sample['curvatures_random'][shuffle, d] = calc_curvature(X_shuffled, degrees=True)

