import random

import torch


class TrainingHyperparameters:
    """
    Hyperparameters for training the model
    """
    def __init__(self, n_epochs, batch_size_train, batch_size_test, learning_rate, momentum,
                 log_interval, random_seed=random.randint(0, 1000000000), cudnn_enabled=False):
        """
        Initialize the hyperparameters
        :param n_epochs: Number of epochs to train
        :param batch_size_train: Batch size for training
        :param batch_size_test: Batch size for testing
        :param learning_rate: Learning rate
        :param momentum: Momentum
        :param log_interval: Number of batches between logging
        :param random_seed: Random seed
        :param cudnn_enabled: Whether to enable cudnn
        """
        self.n_epochs = n_epochs
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.log_interval = log_interval
        self.random_seed = random_seed
        self.cudnn_enabled = cudnn_enabled
        torch.backends.cudnn.enabled = False  # TODO: look up what this does
        torch.manual_seed(self.random_seed)
