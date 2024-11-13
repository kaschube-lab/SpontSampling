from utils import get_args, init_results_dict, update_res_dict, get_save_path
from Curvature import calc_curvature_full
from Dimensionality import calc_dimensionality_increasing_frames
from Entropy import calc_entropy_measures_increasing_frames

def main(args):
    X, area_labels, locations = load_data(args)
    d_results = init_results_dict(len(X), area_labels, locations, args)
    functions = args.functions.split(';')
    save_path = get_save_path(args)
    for function in functions:
        np.random.seed(args.seed)
        update_res_dict(d_results, X, function, args)
        for i, x in enumerate(X):
            n_frames_dt = x.shape[-1] // args.dt
            x = x[:, :int(n_frames_dt*args.dt)]

            if i == 0:
                print('x.shape', x.shape, end=': ')
            if i % 100 == 0:
                print(sample_i, end=',')
            if function.lower() == 'curvature':
                calc_curvature_full(x, d_results[f'sample_{i}'], args)
            elif function.lower() == 'entropy':
                for j in range(args.n_inits):
                    print(j, end=': ', flush=True)
                    calc_entropy_measures_increasing_frames(x, d_results[f'sample_{i}'], args)
                    if j % 5 == 0:
                        # Intermediate save, as the computation takes long and in case of a 
                        # program failure and to analyze intermediate results
                        np.savez_compressed(save_path, **d_results)
            elif function.lower() == 'dimensionality':
                for j in range(args.n_inits):
                    print(j, end=': ', flush=True)
                    calc_dimensionality_increasing_frames(x, d_results[f'sample_{i}'], args)
                    if j % 5 == 0:
                        # Intermediate save, as the computation takes long and in case of a 
                        # program failure and to analyze intermediate results
                        np.savez_compressed(save_path, **d_results)
            
            # Save after each trial/sample/probe
            np.savez_compressed(save_path, **d_results)


if __name__ == '__main__':

    args = get_args()
    main(args)
