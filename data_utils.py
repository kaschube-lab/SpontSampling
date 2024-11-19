import os
import numpy as np
import scipy.io as sio
from skimage.transform import resize

##### ----------------- FERRET ----------------- ######


def get_roi(scale=4):
    roi = np.load('./data/Ferret/common/roi.npy')
    nx, ny = roi.shape
    shape_new = (nx//scale,ny//scale)
    mask_rescaled = resize(roi, output_shape=shape_new, 
                        mode='reflect', order=0)
    return mask_rescaled



# def load_smaller_fov(data_path, size):
#     activity, roi = load_ferret_data(data_path)
#     roi_indices = np.where(roi)
#     max_x, min_x = np.max(roi_indices[0]), np.min(roi_indices[0])
#     max_y, min_y = np.max(roi_indices[1]), np.min(roi_indices[1])
#     center_x = (max_x - min_x) // 2 + min_x
#     center_y = (max_y - min_y) // 2 + min_y
#     square = [np.arange(center_x - size // 2, center_x + size // 2), np.arange(center_y - size // 2, center_y + size // 2)] 
#     act_roi = roi.copy()
#     act_roi[np.where(roi)] = activity
#     act_square = act_roi[square[0]][:, square[1]]
#     assert act_square.shape == (size, size)
#     return act_square.ravel()
    

def load_ferret_data(data_dir='./', EO='before', condition='awake', fDiff=False):
    """
    Loads the ferret data from Trägenap et al. (2023)?
    Parameters:
    -   EO (str): Wheter data is 'before' or 'after' eye opening 
    -   condition (str): Wheter the ferret was 'awake' or anesthetized ('anesth')
    -   fDif (bool): Wheter the data should be preprocessed with first difference calculation.

    Return:
    -   activity (np.array): contains the activity in the form pixels x time frames
    """
    fd = '_FirstDiff' if fDiff else ''
    data_path = os.path.join(data_dir, f'./data/Ferret/{EO}EO/spont/{condition}/DF_by_F0_Resize4x{fd}_bandpass_1_7.5.npy')
    data = np.load(data_path)
    nan_indices = np.where(np.isnan(np.mean(data, axis=0)))[0]

    activity = []
    roi = get_roi()
    print('npixels', len(np.where(roi)[0].ravel()))
    for _, frame in enumerate(data):
        activity.append(frame[np.where(roi)])
    print('act shape', np.array(activity).shape)
    activity = np.array(activity)
    roi[np.where(roi)] *= ~np.isnan(np.mean(activity, axis=0))
    activity = activity[:, np.where(~np.isnan(np.mean(activity, axis=0)))[0]].T
    return activity
    

##### ----------------- HUMAN ----------------- ######



def load_fmri_data(data_dir='./'):
    raw = sio.loadmat(os.path.join(data_dir, "data/Human/NKI_281_Yeo_114.mat"))['subjects']
    # sort_raw_to_orig_indices
    id_raw = np.zeros(281, dtype=int)
    for i in range(281):
        id_raw[i] = int(raw[0, i][0][0].split("A")[1])
    indices = np.argsort(id_raw)

    raw_645 = np.zeros((281, 113, 884))
    for i in range(281):
        raw_645[i, :23] = raw[0, i][3][:23, :884]
        raw_645[i, 23:] = raw[0, i][3][24:, :884]

    raw_zscored = (raw_645 - np.mean(raw_645, axis=-1)[..., np.newaxis]) / np.std(raw_645, axis=-1)[..., np.newaxis]
    raw_zscored = raw_zscored[indices]

    NETWORKS = {
        "VIS": [(0, 4), (56, 60)],
        "SMN": [(5, 9), (61, 65)],
        "DAN": [(10, 16), (66, 72)],
        "VAN": [(17, 26), (73, 85)],
        "LIM": [(27, 28), (86, 87)],
        "CON": [(29, 42), (88, 99)],
        "DMN": [(43, 55), (100, 112)]
    }
    areas = np.zeros(113)
    for i, nw in enumerate(NETWORKS):
        for p in NETWORKS[nw]:
            areas[p[0]:p[1]+1] = i
    
    return raw_zscored, list(NETWORKS.keys()), areas
    

    
    
##### ----------------- MONKEY ----------------- ######



def histc(x, bins):
    map_to_bins = np.digitize(x, bins) # Get indices of the bins to which each value in input array belongs.
    res = np.zeros(bins.shape)
    for el in map_to_bins:
        res[el-1] += 1 # Increment appropriate bin.
    return res
                
    
    
def calc_firing_rate(spike_ts, bin_size, shift):
    end = spike_ts.shape[-1]
    time = 0
    firing_rate = []
    while time <= end - bin_size:
        firing_rate.append(np.sum(spike_ts[:, time:time+bin_size], axis=-1) / bin_size)
        
        time += shift
    
    return np.array(firing_rate)



def preprocess_monkey_ephys_data(bin_size=1000, shift=1):
    all_spike_ts = []
    firing_rates = []
    for i in range(1, 7):
        try:
            monkey_time = sio.loadmat(f'./FerretAE/data/pvc-11/data_and_scripts/spikes_spontaneous/spiketimesmonkey{i}spont.mat')
            monkey_spike_ts = []
            max_event = 0
            for e in range(monkey_time['data']['EVENTS'][0][0].shape[-1]):
                if max_event < monkey_time['data']['EVENTS'][0][0][:, e][0][:, 0][-1]:
                    max_event = monkey_time['data']['EVENTS'][0][0][:, e][0][:, 0][-1]
                
            for e in range(monkey_time['data']['EVENTS'][0][0].shape[-1]):
                print(e, end=', ')
                event_ts = monkey_time['data']['EVENTS'][0][0][:, e][0][:, 0]
                bins = np.arange(0, max_event, .001) # 1 ms bins
                spike_ts = histc(event_ts, bins)
                monkey_spike_ts.append(spike_ts)
            monkey_spike_ts = np.array(monkey_spike_ts)
            all_spike_ts.append(monkey_spike_ts)
            firing_rates.append(calc_firing_rate(monkey_spike_ts, bin_size=bin_size, shift=shift))
            print('')
        except:
            print(i, 'doesnt exist')
    return all_spike_ts, firing_rates


def load_monkey_data(data_dir='./', bin_size=100):
    """
    Loads the ephys data from macaque monkeys 

    Parameters:
    -   bin_size (int): non-overlapping window width to average the spikes within

    Returns: 
    -   data (list): List of neural activations for all trials (shapes: n_neurons, n_timeframes)
    """
    d = np.load(os.path.join(data_dir, f'data/Monkey/pvc_11_{bin_size}ms.npz'))
    data = []
    for i in range(6):
        data.append(d[f'Trial_{i}'].T)
    return data


##### ----------------- STRINGER DATA ----------------- ######

def calc_windowed_spike_matrix(spike_matrix, window_size):
    num_neurons, num_timepoints = spike_matrix.shape
    num_windows = num_timepoints // window_size

    # Reshape and calculate the mean
    spike_matrix_ = spike_matrix[:, :num_windows * window_size].reshape(num_neurons, num_windows, window_size)
    windowed_spike_matrix = spike_matrix_.mean(axis=2)
    return windowed_spike_matrix



def load_stringer_data(animal_name, window_size, data_dir='./'):
    """
    Loads the sponanteous neuropixels data from the stringer paper

    Parameters:
    -   animal_name (str): Name of the mouse (Krebs, Robbins, Wa..)
    -   window_size (int): sliding non-overlapping window width to average the spikes within.

    Returns: 
    -   data_matrix (np.array): Shape XX
    -   area_labels (list): names of possible brain areas that each neuron can belong to
    -   locations (np.array): Indices corresponding to area_labels for each neuron
    """

    f = sio.loadmat(os.path.join(data_dir, f'./data/StringerNeuropixels/{animal_name}withFaces_KS2.mat'))
    data_matrix = calc_windowed_spike_matrix(f['stall'], window_size)
    area_labels = [x[0] for x in f['areaLabels'][0]]
    print('data_matrix.shape', data_matrix.shape)
    return data_matrix, area_labels, f['brainLoc']


##### Load different datasets ######
def load_data(args):
    """
    Parameters:
    -   args (argparse object): arguments
    Returns: 
    -   X ([np.array]): List with data matrices. Shape: (n_samples) x n_neurons/n_rois/ x n_timeframes
    """
    area_labels, locations = None, None
    if args.data_set.lower() == 'stringer': 
        X, area_labels, locations = load_stringer_data(args.animal_name, args.window_size, 
                                                        data_dir=args.data_dir)
        if args.roi:
            try:
                neuron_i = area_labels.index(args.roi) + 1
            except:
                raise ValueError ('Input for roi is not a valid region in the data set', area_labels)
            X = X[np.where(locations == neuron_i)[0]]
            if len(X) == 0:
                raise ValueError (f'Area {args.roi} not in dataset')
            area_labels = None  # they are all the same
            locations = None
        X = [X] # to bring in same format as the other data        

    elif args.data_set.lower() == 'human':
        X, area_labels, locations = load_fmri_data(data_dir=args.data_dir)
        if args.roi:
            try:
                area_i = area_labels.index(args.roi)
            except:
                raise ValueError ('Input for roi is not a valid region in the data set', area_labels)
            X = X[:, np.where(locations == area_i)[0]]
            area_labels = None  # they are all the same
            locations = None
        X = [x for x in X]
    elif args.data_set.lower() == 'ferret':
        X = load_ferret_data(EO=args.EO, condition=args.condition, fDiff=args.FDiff, data_dir=args.data_dir)
        X = [X] # to bring in same format as the other data
    elif args.data_set.lower() == 'monkey':
        X = load_monkey_data(data_dir=args.data_dir)
    else:
        raise ValueError('Input for data_set has to be stringer, human, ferret, or monkey.')
    return X, area_labels, locations