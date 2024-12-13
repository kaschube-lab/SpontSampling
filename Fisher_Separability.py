import numpy as np
from skdim.id import FisherS
from control_data import create_gauss


def calc_fisher_separability(x, d_results_sample, j, args):
    """
    Compute the point-wise fisher separability.

    Args:
        x (np.ndarray): Array of shape (n_pixels, n_timeframes).
        d_results_sample (dict): results dictionary for the current sample
        j (int): current iteration index
        args (argparse object): program arguments

    Returns:
        None
    """

    x = x[:, j::args.dt]
    print(j, 'x.shape', x.shape)
    fishers = FisherS()

    print('compute fisher separability for normal data')
    d_results_sample['fisher_separability'][j] = fishers.fit_transform_pw(x.T)

    print('compute fisher separability for Gauss')
    x_gauss = create_gauss(x)
    d_results_sample['fisher_separability_gauss'][j] = fishers.fit_transform_pw(x_gauss.T)

    print('compute fisher separability for shuffled data')
    for shuffle_i in range(args.n_shuffles):
        x_shuffled = x[:, np.random.permutation(x.shape[-1])]
        d_results_sample['fisher_separability_random'][j] = fishers.fit_transform_pw(x_shuffled)
    

