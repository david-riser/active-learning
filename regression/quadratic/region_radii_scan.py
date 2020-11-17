import os
import pickle
import utils


if __name__ == "__main__":

    config = dict(
        n_samples = 10000,
        n_init_pts = 10,
        n_iter = 24,
        n_trials = 64,
        n_test_samples = 1000,
        n_dev_samples = 1000,
        input_dims = 2,
        n_layers = 3,
        n_nodes = 16,
        batches = 1000,
        batch_size = 32,
        lr = 0.01
    )

    data_bundle = utils.setup_data(config)
    radii = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4]
    experiments = []

    for radius in radii:
        
        trials = utils.easy_distribute(
            func = lambda: utils.run_region_experiment(config, data_bundle, region_radius=radius),
            n_trials = config["n_trials"],n_cores = os.cpu_count())

        experiments.append(trials)
        
    results = dict(
        experiments = experiments,
        config = config,
        radii = radii
    )

    with open("region_radius_scan_{}d.pkl".format(config["input_dims"]), "wb") as dumper:
        pickle.dump(results, dumper)
