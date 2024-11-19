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


def get_nearest_neighbours(x_norm):
    """
    Compute the nearest neightbours and cosine similarity between each time point and all previous time points.
    Parameters:
    - x_norm (np.array): normalized input matrix of shape (n_neurons, n_timeframes).
    
    Returns:
    - nn (np.array): ordering of previous time points by cosine similarity of shape (n_timeframes, n_timeframes)
    - cosine_similarities (np.array): cos sims of shape (n_timeframes, n_timeframes)
    """
    _ , n_timeframes = x_norm.shape
    nn, cosine_similarities = np.zeros((n_timeframes, n_timeframes)), np.zeros((n_timeframes, n_timeframes))
    for t in range(1, n_timeframes):
        cos = np.dot(x_norm[:, t].reshape(-1), x_norm[:, :t])
        cosine_similarities[t, :t] = cos
        nn[t, :t] = np.argsort(cos)
    return nn, cosine_similarities


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
    
    start_frame = np.random.randint(n_timeframes // args.dt - (args.steps + args.min_frames))
    X_subset = x[:, start_frame::args.dt][:, :args.steps]


    # Normalize the matrix along the neurons dimension
    X_subset = X_subset / np.linalg.norm(X_subset, axis=0, keepdims=True)

    # Compute the neighbourhood order and cosine similarity for each time frame    
    nn, cosine_similarity = get_nearest_neighbours(X_subset)
    d_results[f'NN'][j] = nn
    d_results[f'Cosine_similarity'][j] = cosine_similarity

    for shuffle_i in range(args.n_shuffles):
        X_shuffled = np.apply_along_axis(np.random.permutation, 1, X_subset)
        nn, cosine_similarity = get_nearest_neighbours(X_shuffled)
        d_results[f'NN_random'][j] = nn
        d_results[f'Cosine_similarity_random'][shuffle_i, j] = cosine_similarity

    
