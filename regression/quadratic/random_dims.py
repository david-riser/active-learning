import os
import pickle
import utils


if __name__ == "__main__":

    dims = [2, 4, 6, 8]
    pts = [10, 100, 100, 1000]
    results = [] 
    
    for dim, points in zip(dims,pts):

        config = dict(
            n_samples = 10000,
            n_init_pts = points,
            n_iter = 16,
            n_trials = 32,
            n_test_samples = 1000,
            n_dev_samples = 1000,
            input_dims = dim,
            n_layers = 3,
            n_nodes = 16,
            batches = 1000,
            batch_size = 32,
            lr = 0.01
        )

        data_bundle = utils.setup_data(config)
        trials = utils.easy_distribute(
            func = lambda: utils.run_random_experiment(config, data_bundle),
            n_trials = config["n_trials"],
            n_cores = os.cpu_count()
        )

        results.append(trials)
        
    results_dict = dict(
        trials = results,
        config = config 
    )

    with open("compare_dimensions_random.pkl", "wb") as dumper:
        pickle.dump(results_dict, dumper)
