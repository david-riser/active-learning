import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


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


def worker(queue, x_train, y_train, x_dev, y_dev, x_test, y_test,
           labeled_pts, unlabeled_pts, n_init_pts):

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
            x_dev=x_dev, y_dev=y_dev, batches=12, batch_size=64, eval_freq=10)
        
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

    
if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_data()
    n_test_samples = x_test.shape[0]

    
    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_test = x_test.reshape(len(x_test), 1, 28, 28)

    n_init_pts = 100
    n_iter = 3
    n_trials = 6
    
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


    do_work = lambda q: worker(q, x_train, y_train, x_dev, y_dev, x_test, y_test,
                             labeled_pts, unlabeled_pts, n_init_pts)

    

    n_cores = os.cpu_count() 

    queue = Queue()
    workers = []

    for job in range(n_trials):
        workers.append(Process(target=do_work, args=(queue,)))
        workers[job].start()

    result_pool = []
    for worker in workers:
        result_pool.append(queue.get())

    for worker in workers:
        worker.join()


    print(result_pool)
