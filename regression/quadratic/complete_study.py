"""
    The complete study.  Everything is compared back to the random control. 

    Two sizes are compared, a low dimensional and a higher dimensional case (which 
    still isn't really high dimensional).  

    1) A random trial is performed for each dimensionality
    2) An uncertainty sampling trial is performed for each dimensionality 
    3) A region sampling trial is performed for each radius in each dimensionality
    
    Results are saved to a pickle file for visualization.

"""
import os
import pickle
import utils


if __name__ == "__main__":

    config = dict(
        n_samples = 10000,
        n_init_pts = 10,
        n_iter = 24,
        n_trials = 32,
        n_test_samples = 1000,
        n_dev_samples = 1000,
        input_dims = 2,
        n_layers = 3,
        n_nodes = 16,
        batches = 1000,
        batch_size = 32,
        lr = 0.01
    )

    # Parameter sweep setup 
    radii = [0.01, 0.1, 0.2, 0.4, 1.0]
    n_dimensions = [2, 6]

    # Setup data bundles for the different dimensions
    # that we want to try in our experiments. 
    data_bundles = {}
    for input_dims in n_dimensions:
        config["input_dims"] = input_dims
        data_bundles[input_dims] = utils.setup_data(config)
    

    # Take a random baseline for each dimension
    random_controls = {}
    print("[CompleteStudy] Running random control for each data bundle...")
    for input_dims in n_dimensions:
        config["input_dims"] = input_dims
  
        random_controls[input_dims] = utils.easy_distribute(
            func = lambda: utils.run_random_experiment(config, data_bundles[input_dims]),
            n_trials = config["n_trials"], n_cores = os.cpu_count()
        )
        

    # Uncertainty sampling for each dimension 
    uncertainty_experiments = {} 
    print("[CompleteStudy] Running uncertainty sampling for each data bundle...")
    for input_dims in n_dimensions:
        config["input_dims"] = input_dims
  
        uncertainty_experiments[input_dims] = utils.easy_distribute(
            func = lambda: utils.run_uncertainty_experiment(config, data_bundles[input_dims]),
            n_trials = config["n_trials"], n_cores = os.cpu_count()
        )


    # Region sampling for each dimension and radius 
    region_experiments = {} 
    print("[CompleteStudy] Running uncertainty sampling for each data bundle...")
    for input_dims in n_dimensions:
        config["input_dims"] = input_dims

        region_experiments[input_dims] = []

        for radius in radii:
            region_experiments[input_dims].append(
                utils.easy_distribute(
                    func = lambda: utils.run_region_experiment(config, data_bundles[input_dims], region_radius=radius),
                    n_trials = config["n_trials"], n_cores = os.cpu_count()
                )
            )
        

        
    results = dict(
        random_controls = random_controls,
        region_experiments = region_experiments,
        uncertainty_experiments = uncertainty_experiments,
        input_dims = n_dimensions, 
        config = config,
        radii = radii
    )

    
    with open("complete_study.pkl".format(config["input_dims"]), "wb") as dumper:
        pickle.dump(results, dumper)
