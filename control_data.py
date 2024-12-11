import numpy as np
import os
from scipy.stats import multivariate_normal

# Create control data

def create_gauss(X):
    """
    Method to create control data sampled from a gaussian distribution based on covariance of data
    Parametres:
        X (np.array): data array with shape (n_pixels, n_time_frames)

    Returns:
    - Y (np.array): synthetically created control data in the same shape as X
    """
    _, n_timeframes = X.shape
    # Compute the mean vector (average over timepoints for each variable)
    mean = np.mean(X, axis=1)
    # Compute the covariance matrix (pairwise covariance between variables)
    covariance = np.cov(X)

    # Create a multivariate normal distribution object
    mvn = multivariate_normal(mean=mean, cov=covariance, allow_singular=True)

    Y = mvn.rvs(size=n_timeframes).T  # Transpose to match (n_pixels, n_timeframes)
    return Y
    

def load_movie(data_dir='./', movie=1):
    # Potential ToDo: Load all movies into the list
    if movie == 1:
        folder_path = os.path.join(data_dir, "AllenInstitute_Movies/natural_movie_one.npy")
    elif movie == 3:
        folder_path = os.path.join(data_dir, "AllenInstitute_Movies/natural_movie_three.npy")

    movie_data = np.load(folder_path)
    movie_flat = movie_data.reshape(movie_data.shape[0],-1).T  # reshape to fit (n_pixels, n_timeframes)

    return [movie_flat]