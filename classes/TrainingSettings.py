class TrainingSettings:
    def __init__(self, network, hyperparams, optimizer=None, train_loader=None, test_loader=None):
        self.network = network
        self.hyperparams = hyperparams
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
