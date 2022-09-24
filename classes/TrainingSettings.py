class TrainingSettings:
    """
    All settings for training a network
    """
    def __init__(self, network, hyperparams, optimizer=None, train_loader=None, test_loader=None):
        """
        Initialize the training settings
        :param network: The network architecture to train
        :param hyperparams: The hyperparameters to use for training
        :param optimizer: The optimizer to use for training
        :param train_loader: The training data loader
        :param test_loader: The testing data loader
        """
        self.network = network
        self.hyperparams = hyperparams
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
