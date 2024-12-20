import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_cosine_save_path(args):
    res_dir = os.path.join(args.save_dir, 'results', 'Cosine')
    os.makedirs(res_dir, exist_ok=True)
    save_name = f'Cosine_{args.data_set}_{args.dt}dt'
    if args.data_set.lower() == 'stringer':
        save_name += f'_{args.animal_name}_{args.window_size}windowsize'
    elif args.data_set.lower() == 'ferret':
        fd = 'FDiff' if args.FDiff else 'Orig'
        save_name += f'{args.EO}EO_{args.condition}_{fd}'
    if args.roi:
        save_name +=  f'_{args.roi}' 
    save_name += 'npy'
    save_path = os.path.join(res_dir, save_name)
    return save_path


def get_nearest_neighbours(x_norm, d_nn, d_cos, args):
    """
    Compute the nearest neightbours and cosine similarity between each time point and all previous time points.
    Parameters:
    - x_norm (np.array): normalized input matrix of shape (n_neurons, n_timeframes).
    
    Returns:
    - nn (np.array): ordering of previous time points by cosine similarity of shape (n_timeframes, n_timeframes)
    - cosine_similarities (np.array): cos sims of shape (n_timeframes, n_timeframes)
    """
    _ , n_timeframes = x_norm.shape
    # nn, cosine_similarities = np.zeros((n_timeframes, n_timeframes)), np.zeros((n_timeframes, n_timeframes))
    for t in range(1, n_timeframes):
        cos = np.dot(x_norm[:, t].reshape(-1), x_norm[:, :t])
        d_cos[t, :t] = cos
        if t >= args.min_frames:
            d_nn[t-args.min_frames] = np.argsort(cos[-args.min_frames:])



def calc_kNN(x, d_results, j, args):
    """
    Compute the k-nearest neighbours of each time frame to all previous time frames. As a distance matrix
    we use the cosine similarity. 

    Parameters:
    - x (np.ndarray): The input data matrix with shape (n_neurons, n_timeframes).
    - d_results (dict): results dictionalry
    - j (int): current iteration index
    - args (argparse object): program arguments

    Returns:
    - None
    """
    n_neurons, n_timeframes = x.shape
    
    #start_frame = np.random.randint(n_timeframes // args.dt - (args.steps + args.min_frames))
    x = x[:, j::args.dt]

    norm = np.linalg.norm(x, axis=0, keepdims=True)
    norm = np.where(norm < args.knn_epsilon, 0, norm)

    # Normalize the matrix along the neurons dimension
    x = x / norm
    # x[np.isnan(x)] = np.nan

    # Compute the neighbourhood order and cosine similarity for each time frame    
    get_nearest_neighbours(x, d_results[f'NN'][j], d_results[f'Cosine_similarity'][j], args)
    #d_results[f'NN'][j] = nn
    #d_results[f'Cosine_similarity'][j] = cosine_similarity

    for shuffle_i in range(args.n_shuffles):
        X_shuffled = np.apply_along_axis(np.random.permutation, 1, x)
        get_nearest_neighbours(X_shuffled, d_results[f'NN_random'][shuffle_i, j], 
                                d_results[f'Cosine_similarity_random'][shuffle_i, j], args)
        # d_results[f'NN_random'][shuffle_i, j] = nn
        # d_results[f'Cosine_similarity_random'][shuffle_i, j] = cosine_similarity

    
