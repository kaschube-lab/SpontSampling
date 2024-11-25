import numpy as np
import os
from scipy.stats import multivariate_normal

# Create control data

def create_gauss(X, n_samples):
    """
    Method to create control data sampled from a gaussian distribution based on covariance of data
    Parametres:
        X (np.array): data array with shape (number_pixels, number_time_frames)
        n_samples (int): number of synthetic data points

    Returns:
    - Y (np.array): synthetically created control data
    """

    # Compute the mean vector (average over timepoints for each variable)
    mean = np.mean(X, axis=1)
    # Compute the covariance matrix (pairwise covariance between variables)
    covariance = np.cov(X)

    # Create a multivariate normal distribution object
    mvn = multivariate_normal(mean=mean, cov=covariance)

    Y = mvn.rvs(size=n_samples).T  # Transpose to match (number_pixels, n_samples)

    return Y


def load_movie(data_dir='./', movie=1):
    if movie == 1:
        folder_path = os.path.join(data_dir, "AllenInstitute_Movies/natural_movie_one.npy")
    elif movie == 3:
        folder_path = os.path.join(data_dir, "AllenInstitute_Movies/natural_movie_three.npy")

    movie_data = np.load(folder_path)
    movie_flat = movie_data.reshape(movie_data.shape[0],-1).T  # reshape to fit (n_pixels, n_timeframes)

    return movie_flat