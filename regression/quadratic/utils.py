import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
from multiprocessing import Process, Queue

DataBundle = namedtuple("DataBundle", "x_train y_train x_dev y_dev x_test y_test index_pool first_pts")

def setup_data(config):
    x_train = np.random.uniform(-3, 3, (config["n_samples"], config["input_dims"]))
    y_train = target_function(x_train)
    x_test = np.random.uniform(-3, 3, (config["n_test_samples"], config["input_dims"]))
    y_test = target_function(x_test)
    x_dev = np.random.uniform(-3, 3, (config["n_dev_samples"], config["input_dims"]))
    y_dev = target_function(x_dev)

    index_pool = np.random.permutation(np.arange(len(x_train)))
    first_pts = np.random.choice(a=index_pool, size=config["n_init_pts"], replace=False)
    return DataBundle(x_train, y_train, x_dev, y_dev, x_test, y_test, index_pool, first_pts)



def target_function(x):
    return 0.1 * np.sum(x**2, axis=1)


class Net(nn.Module):
    def __init__(self, n_nodes, n_layers, input_dims):
        super(Net, self).__init__()

        self.input_layer = nn.Sequential(nn.Linear(input_dims, n_nodes), nn.Sigmoid())
        self.output_layer = nn.Linear(n_nodes, 1)

        self.fc_layers = [nn.Sequential(nn.Linear(n_nodes, n_nodes), nn.Sigmoid(), nn.Dropout(0.2))
            for _ in range(n_layers)
        ]
        self.fc_layers = nn.Sequential(*self.fc_layers)

        self.features = nn.Sequential(
            self.input_layer, 
            self.fc_layers, 
            self.output_layer
        )

    def forward(self, x):
        return self.features(x)


def train_network(network, opt, x_train, y_train, 
                  x_dev, y_dev, batches, batch_size, 
                  eval_freq, fuzz_factor=0.0):


    x_dev_ = torch.FloatTensor(x_dev)
    y_dev_ = torch.FloatTensor(y_dev).reshape(len(y_dev),1)
    index_pool = np.arange(len(x_train))
    best_model = None
    best_loss = np.inf
    train_loss = []
    dev_loss = []
    for batch in range(batches):
        indices = np.random.choice(index_pool, batch_size)
        x_batch = torch.FloatTensor(x_train[indices])
        y_batch = torch.FloatTensor(y_train[indices]).reshape((batch_size,1))
        
        if fuzz_factor > 0.0:
            x_batch += torch.normal(mean=0., std=fuzz_factor, size=x_batch.shape)
        
        y_batch_pred = network(x_batch)

        opt.zero_grad()
        batch_loss = F.mse_loss(y_batch_pred, y_batch)
        batch_loss.backward() 
        opt.step()

        train_loss.append(batch_loss.detach().numpy())
        
        if batch % 10 == 0 and batch > 0:
            with torch.no_grad():
                y_pred_dev = network(x_dev_)
                d_loss = F.mse_loss(y_pred_dev, y_dev_)
                dev_loss.append(d_loss)
                if d_loss < best_loss:
                    best_model = network.state_dict() 
                    best_loss = d_loss

    network.load_state_dict(best_model)
    return network, train_loss, dev_loss



def uncertainty_sample(network, x_train, unlabeled_pts, n_pts, 
                       batch_size=8, bayes_samples=64):
    pool = list(unlabeled_pts)
    batches = len(pool) // batch_size
    output = []

    for batch in range(batches):
        indices = pool[batch * batch_size : batch * batch_size + batch_size]
        samples = []
        for sample in range(bayes_samples):
            inputs = torch.FloatTensor(x_train[indices])
            pred = network(inputs)
            samples.append(pred)

        vote_entropy = torch.std(torch.stack(samples)).detach().numpy()
        output.append(vote_entropy)

    ordering = np.argsort(output)[::-1]
    return [pool[index] for index in ordering[:n_pts]]


def region_sample(network, x_train, y_train, region_radius, n_pts):
    """ Sample inside of a region around the worst points.  This should
        not have access to the entire training set, just the labeled
        portion of it!
    """
    
    loss = np.zeros(x_train.shape[0])
    test_batch_size = 100
    test_batches = x_train.shape[0] // test_batch_size
    for test_batch in range(test_batches):
        start = test_batch * test_batch_size
        stop = start + test_batch_size
        y_pred = network(torch.FloatTensor(x_train[start:stop]))
        y_pred = y_pred.detach().numpy() 
        residual = (y_pred.squeeze() - y_train[start:stop])**2
        loss[start:stop] = residual

    ordering = np.argsort(loss)[::-1]
    
    n_samples, n_features = x_train.shape
    x_new = np.zeros((n_pts, n_features))

    for i in range(n_pts):
        x_new[i] = np.random.normal(0., region_radius) + x_train[ordering[i]]
        
    y_new = target_function(x_new)
    return x_new, y_new


def softmax_temperature(x, temperature):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


def region_sample2(network, x_train, y_train, region_radius, n_pts, temperature):
    """ Sample inside of a region around the worst points. """
    
    loss = np.zeros(x_train.shape[0])
    test_batch_size = 100
    test_batches = x_train.shape[0] // test_batch_size
    for test_batch in range(test_batches):
        start = test_batch * test_batch_size
        stop = start + test_batch_size
        y_pred = network(torch.FloatTensor(x_train[start:stop]))
        y_pred = y_pred.detach().numpy() 
        residual = (y_pred.squeeze() - y_train[start:stop])**2
        loss[start:stop] = residual

    loss = softmax_temperature(loss, temperature)
    ordering = np.argsort(loss)[::-1]
    
    n_samples, n_features = x_train.shape
    x_new = np.zeros((n_pts, n_features))

    for i in range(n_pts):
        x_new[i] = np.random.normal(0., region_radius) + x_train[ordering[i]]
        
    y_new = target_function(x_new)
    return x_new, y_new



# The "experiment" is a series of random trials over the same
# set of initial point.  Each trial proceeds in a number of 
# training iterations. 

def perform_training_iteration(network_config, x_train, y_train, 
                               x_dev, y_dev, x_test, y_test, test_batch_size,
                               eval_freq):
    """ Perform one training iteration. """
    
    # Setup network and train
    network = Net(
        n_nodes=network_config["n_nodes"], 
        n_layers=network_config["n_layers"], 
        input_dims=network_config["input_dims"]
    )
    opt = optim.Adam(params=network.parameters(), 
                     lr=network_config["lr"])

    network, train_loss, dev_loss = train_network(
        network=network, opt=opt, x_train=x_train, y_train=y_train, 
        x_dev=x_dev, y_dev=y_dev, 
        batches=network_config["batches"], 
        batch_size=network_config["batch_size"], 
        eval_freq=eval_freq
    )

        
    # Evaluate the test set.
    test_batches = len(x_test) // test_batch_size
    test_loss = 0.
    for test_batch in range(test_batches):
        start = test_batch * test_batch_size
        stop = start + test_batch_size
        y_test_pred = network(torch.FloatTensor(x_test[start:stop]))
        test_loss += F.mse_loss(y_test_pred, torch.FloatTensor(y_test[start:stop]).reshape((stop-start,1))) / test_batches
        
    return test_loss, network


def easy_distribute(func, n_trials, n_cores):
    """ Distribute the jobs required by this function over 
        the processors.
    """

    def wrapped_work(queue):
        result = func()
        queue.put(result)

    queue = Queue() 
    workers = [] 
    for job in range(n_trials):
        workers.append(Process(target=wrapped_work, args=(queue,)))

    result_pool = []
    batches = int(np.ceil(n_trials / n_cores))
    for batch in range(batches):
        batch_jobs = n_cores
        if (batch == batches - 1) and n_trials % n_cores != 0:
            batch_jobs = n_trials % n_cores
        
        print("Starting batch {} with {} jobs.".format(batch, batch_jobs))
        
        for core in range(batch_jobs):
            workers[batch * n_cores + core].start()

        for core in range(batch_jobs):
            result_pool.append(queue.get())

        for core in range(batch_jobs):
            workers[batch * n_cores + core].join()

    return np.stack(result_pool)


def run_random_experiment(config, data_bundle):

    np.random.seed(os.getpid())
    torch.manual_seed(os.getpid())

    trials = np.zeros(config["n_iter"])

    labeled_pts = set(data_bundle.first_pts)
    unlabeled_pts = set(data_bundle.index_pool)
    for point in data_bundle.first_pts:
        unlabeled_pts.remove(point)

    for iter in range(config["n_iter"]):
                
        # Add n_init_pts to labeled randomly. 
        if iter > 0:
            for point in np.random.choice(a=list(unlabeled_pts), size=config["n_init_pts"], replace=False):
                unlabeled_pts.remove(point)
                labeled_pts.add(point)

        # Setup training set for this round.
        x_train_ = data_bundle.x_train[list(labeled_pts)]
        y_train_ = data_bundle.y_train[list(labeled_pts)]

        test_loss, network = perform_training_iteration(network_config=config, x_train=x_train_,
                                y_train=y_train_, x_dev=data_bundle.x_dev, y_dev=data_bundle.y_dev,
                                x_test=data_bundle.x_test, y_test=data_bundle.y_test, test_batch_size=100, 
                                eval_freq=1)
        trials[iter] = test_loss

    return trials


def run_uncertainty_experiment(config, data_bundle,
                          sample_batch_size=32, bayes_samples=16):
    
    np.random.seed(os.getpid())
    torch.manual_seed(os.getpid())
    trials = np.zeros(config["n_iter"])
        
    labeled_pts = set(data_bundle.first_pts)
    unlabeled_pts = set(data_bundle.index_pool)
    for point in data_bundle.first_pts:
        unlabeled_pts.remove(point)

    for iter in range(config["n_iter"]):
            
        # Add n_init_pts to labeled randomly. 
        if iter > 0:
            new_pts = uncertainty_sample(network, data_bundle.x_train, 
                            list(unlabeled_pts), n_pts=config["n_init_pts"], 
                            batch_size=sample_batch_size, bayes_samples=bayes_samples)
            for point in new_pts:
                if point in unlabeled_pts:
                    unlabeled_pts.remove(point)
                labeled_pts.add(point)
                
        # Setup training set for this round.
        x_train_ = data_bundle.x_train[list(labeled_pts)]
        y_train_ = data_bundle.y_train[list(labeled_pts)]

        test_loss, network = perform_training_iteration(network_config=config, x_train=x_train_,
                                y_train=y_train_, x_dev=data_bundle.x_dev, y_dev=data_bundle.y_dev,
                                x_test=data_bundle.x_test, y_test=data_bundle.y_test, test_batch_size=100, 
                                eval_freq=1)
        trials[iter] = test_loss

    return trials


def run_region_experiment(config, data_bundle, region_radius=0.02):
    
    np.random.seed(os.getpid())
    torch.manual_seed(os.getpid())
    trials = np.zeros(config["n_iter"])
        
    labeled_pts = set(data_bundle.first_pts)
    unlabeled_pts = set(data_bundle.index_pool)
    for point in data_bundle.first_pts:
        unlabeled_pts.remove(point)

    x_train_ = data_bundle.x_train[list(labeled_pts)]
    y_train_ = data_bundle.y_train[list(labeled_pts)]
    for iter in range(config["n_iter"]):
            
        # Add n_init_pts to labeled randomly. 
        if iter > 0:
            x_new, y_new = region_sample(network, x_train_, y_train_, 
                                        n_pts=config["n_init_pts"], region_radius=region_radius)
            
            # Setup training set for this round.
            x_train_ = np.concatenate([x_train_, x_new])
            y_train_ = np.concatenate([y_train_, y_new])

            
        test_loss, network = perform_training_iteration(network_config=config, x_train=x_train_,
                                y_train=y_train_, x_dev=data_bundle.x_dev, y_dev=data_bundle.y_dev,
                                x_test=data_bundle.x_test, y_test=data_bundle.y_test, test_batch_size=100, 
                                eval_freq=1)
        trials[iter] = test_loss

    return trials


def run_region2_experiment(config, data_bundle, 
                          temperature=1., region_radius=0.02):
    
    np.random.seed(os.getpid())
    torch.manual_seed(os.getpid())
    trials = np.zeros(config["n_iter"])
        
    labeled_pts = set(data_bundle.first_pts)
    unlabeled_pts = set(data_bundle.index_pool)
    for point in data_bundle.first_pts:
        unlabeled_pts.remove(point)

    x_train_ = data_bundle.x_train[list(labeled_pts)]
    y_train_ = data_bundle.y_train[list(labeled_pts)]
    for iter in range(config["n_iter"]):
            
        # Add n_init_pts to labeled randomly. 
        if iter > 0:
            x_new, y_new = region_sample2(network, x_train_, y_train_, 
                                        n_pts=config["n_init_pts"], region_radius=region_radius, 
                                        temperature=temperature)
            
            # Setup training set for this round.
            x_train_ = np.concatenate([x_train_, x_new])
            y_train_ = np.concatenate([y_train_, y_new])

        test_loss, network = perform_training_iteration(network_config=config, x_train=x_train_,
                                y_train=y_train_, x_dev=data_bundle.x_dev, y_dev=data_bundle.y_dev,
                                x_test=data_bundle.x_test, y_test=data_bundle.y_test, test_batch_size=100, 
                                eval_freq=1)
        trials[iter] = test_loss

    return trials

