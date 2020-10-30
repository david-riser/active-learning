import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from copy import copy
from multiprocessing import Process, Queue
from tensorflow.keras.datasets.mnist import load_data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def acc(y_batch_pred, y_batch):
    y_pred_class = torch.argmax(y_batch_pred, dim=1)
    correct = y_pred_class.eq(y_batch.view_as(y_pred_class)).sum().item()
    return float(correct / len(y_pred_class))


def train_network(network, opt, x_train, y_train, 
                  x_dev, y_dev, batches, batch_size, 
                  eval_freq):


    x_dev_ = torch.FloatTensor(x_dev)
    y_dev_ = torch.LongTensor(y_dev)
    index_pool = np.arange(len(x_train))
    best_model = None
    best_acc = 0.
    train_loss = []
    train_acc = [] 
    dev_acc = []
    for batch in range(batches):
        indices = np.random.choice(index_pool, batch_size)
        x_batch = torch.FloatTensor(x_train[indices])
        y_batch = torch.LongTensor(y_train[indices])
        y_batch_pred = network(x_batch)

        opt.zero_grad()
        batch_loss = F.nll_loss(y_batch_pred, y_batch)
        batch_loss.backward() 
        opt.step()

        train_acc.append(acc(y_batch_pred, y_batch))
        train_loss.append(batch_loss.detach().numpy())
        
        if batch % 10 == 0 and batch > 0:
            with torch.no_grad():
                y_pred_dev = network(x_dev_)
                accuracy = acc(y_pred_dev, y_dev_)
                dev_acc.append(accuracy)
                if accuracy > best_acc:
                    best_model = network.state_dict() 
                    best_acc = accuracy

    network.load_state_dict(best_model)
    return network, train_loss, train_acc, dev_acc


def random_worker(queue, x_train, y_train, x_dev, y_dev, x_test, y_test,
                  labeled_pts, unlabeled_pts, n_init_pts):

    np.random.seed(os.getpid())
    torch.manual_seed(os.getpid())


    print("Working...")
    
    trials = np.zeros(n_iter)
    for iter in range(n_iter):
        
        print("Iter {}".format(iter))

        # Add n_init_pts to labeled randomly. 
        if iter > 0:
            for point in np.random.choice(a=list(unlabeled_pts), size=n_init_pts, replace=False):
                unlabeled_pts.remove(point)
                labeled_pts.add(point)

        # Setup training set for this round.
        x_train_ = x_train[list(labeled_pts)]
        y_train_ = y_train[list(labeled_pts)]
        
        # Setup network and train
        network = Net()
        opt = optim.Adam(params=network.parameters(), lr=0.001)

        network, train_loss, train_acc, dev_acc = train_network(
            network=network, opt=opt, x_train=x_train_, y_train=y_train_, 
            x_dev=x_dev, y_dev=y_dev, batches=1500, batch_size=64, eval_freq=10)
        
        # Evaluate the test set.
        test_batch_size = 100
        test_batches = n_test_samples // test_batch_size
        test_acc = 0.
        for test_batch in range(test_batches):
            start = test_batch * test_batch_size
            stop = start + test_batch_size
            y_test_pred = network(torch.FloatTensor(x_test[start:stop]))
            test_acc += acc(y_test_pred, torch.LongTensor(y_test[start:stop])) / test_batches

        trials[iter] = test_acc

    print("Done work.")
    queue.put(trials)


    
def uncertainty_worker(queue, x_train, y_train, x_dev, y_dev, x_test, y_test,
                       labeled_pts, unlabeled_pts, n_init_pts):

    np.random.seed(os.getpid())
    torch.manual_seed(os.getpid())

    print("Working...")

    unlabeled_pts_ = copy(unlabeled_pts)
    labeled_pts_ = copy(labeled_pts)
    
    trials = np.zeros(n_iter)
    for iter in range(n_iter):
        
        print("Iter {}".format(iter))

        # Add n_init_pts to labeled randomly. 
        if iter > 0:
            print("Sampling new points..")
            subset = np.random.choice(list(unlabeled_pts_), 1024)
            new_pts = uncertainty_sample(network, x_train, subset, n_pts=n_init_pts,
                                         batch_size=64, bayes_samples=8)
            
            for point in new_pts:
                if point in unlabeled_pts_:
                    unlabeled_pts_.remove(point)
                else:
                    print("Troubling finding {} in unlabed...".format(point))
                labeled_pts_.add(point)
            

        # Setup training set for this round.
        x_train_ = x_train[list(labeled_pts_)]
        y_train_ = y_train[list(labeled_pts_)]
        
        # Setup network and train
        print("Training network...")
        network = Net()
        opt = optim.Adam(params=network.parameters(), lr=0.001)

        network, train_loss, train_acc, dev_acc = train_network(
            network=network, opt=opt, x_train=x_train_, y_train=y_train_, 
            x_dev=x_dev, y_dev=y_dev, batches=1500, batch_size=64, eval_freq=10)
        
        # Evaluate the test set.
        print("Evaluating for scores...")
        test_batch_size = 100
        test_batches = n_test_samples // test_batch_size
        test_acc = 0.
        for test_batch in range(test_batches):
            start = test_batch * test_batch_size
            stop = start + test_batch_size
            y_test_pred = network(torch.FloatTensor(x_test[start:stop]))
            test_acc += acc(y_test_pred, torch.LongTensor(y_test[start:stop])) / test_batches

        trials[iter] = test_acc

    print("Done work.")
    queue.put(trials)



def uncertainty_sample(network, x_train, unlabeled_pts, n_pts, 
                       batch_size=8, bayes_samples=64):
    pool = list(unlabeled_pts)
    batches = len(pool) // batch_size
    output = []

    for batch in range(batches):
        print("Predicting batch {}/{}".format(batch, batches))
        indices = pool[batch * batch_size : batch * batch_size + batch_size]
        samples = []
        for sample in range(bayes_samples):
            inputs = torch.FloatTensor(x_train[indices])
            pred = network(inputs)
            samples.append(pred)

        probs = torch.exp(torch.stack(samples).mean(axis=0))
        vote_entropy = torch.sum(-1 * torch.log(probs) * probs, axis=1).detach().numpy()
        output.append(vote_entropy)

    votes = []
    for probs in output:
        votes.extend(probs)

    ordering = np.argsort(votes)[::-1]
    return [pool[index] for index in ordering[:n_pts]]



if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_data()
    n_test_samples = x_test.shape[0]

    
    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_test = x_test.reshape(len(x_test), 1, 28, 28)

    n_init_pts = 100
    n_iter = 16
    n_trials = 32
    
    index_pool = np.random.permutation(np.arange(len(x_train)))
    breakpoint = int(0.8 * len(index_pool))
    
    train_indices = index_pool[:breakpoint]
    dev_indices = index_pool[breakpoint:]
    x_dev = x_train[dev_indices]
    y_dev = y_train[dev_indices]
    
    trials = np.zeros((n_trials, n_iter))
    first_pts = np.random.choice(a=train_indices, size=n_init_pts, replace=False)

    labeled_pts = set(first_pts)
    unlabeled_pts = set(train_indices)
    for point in first_pts:
        unlabeled_pts.remove(point)

    n_cores = os.cpu_count() 
    queue = Queue()
    workers = []

    for job in range(n_trials):
        workers.append(Process(target=random_worker, args=(queue,x_train, y_train, x_dev, y_dev, x_test, y_test,
                             labeled_pts, unlabeled_pts, n_init_pts)))


    random_result_pool = []

    batches = int(np.ceil(n_trials / n_cores))
    for batch in range(batches):
        batch_jobs = n_cores
        if (batch == batches - 1) and n_trials % n_cores != 0:
            batch_jobs = n_trials % n_cores
            
        print("Starting batch {} with {} jobs.".format(batch, batch_jobs))
        
        for core in range(batch_jobs):
            workers[batch * n_cores + core].start()

        for core in range(batch_jobs):
            random_result_pool.append(queue.get())

        for core in range(batch_jobs):
            workers[batch * n_cores + core].join()
            
    unc_result_pool = [] 
    queue = Queue()
    workers = []

    for job in range(n_trials):
        workers.append(Process(target=uncertainty_worker, args=(queue,x_train, y_train, x_dev, y_dev, x_test, y_test,
                             labeled_pts, unlabeled_pts, n_init_pts)))
        
    for batch in range(batches):
        batch_jobs = n_cores
        if (batch == batches - 1) and n_trials % n_cores != 0:
            batch_jobs = n_trials % n_cores
            
        print("Starting batch {} with {} jobs.".format(batch, batch_jobs))
        
        for core in range(batch_jobs):
            workers[batch * n_cores + core].start()

        for core in range(batch_jobs):
            unc_result_pool.append(queue.get())

        for core in range(batch_jobs):
            workers[batch * n_cores + core].join()
            

    results = {"random":random_result_pool, "uncertainty":unc_result_pool}
    with open("results.pkl", "wb") as outf:
        pickle.dump(results, outf)
