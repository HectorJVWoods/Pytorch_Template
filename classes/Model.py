from helpers import trainer


class Model:
    """
    A model prior to training.
    """
    def __init__(self, name, net, hyperparams, optimizer, training_settings,
                 train_loader, test_loader, img_transform=None):
        """
        Create a new model.
        :param name: The name of the model. The directory where the model will be saved will be named this.
        (as a subdirectory of the results directory)
        :param net: Network architecture. Must be a subclass of torch.nn.Module.
        :param hyperparams:  Hyperparameters for the network. Instance of the TrainingHyperparameters class.
        :param optimizer: Optimizer to use. Must be a subclass of torch.optim.
        :param training_settings: Training settings. Instance of the TrainingSettings class.
        :param train_loader: Training data loader. Must be a subclass of torch.utils.data.DataLoader.
        :param test_loader: Test data loader. Must be a subclass of torch.utils.data.DataLoader.
        :param img_transform: Image transform to use. Must be a subclass of torchvision.transforms.
        """
        self.name = name
        self.net = net
        self.hyperparams = hyperparams
        self.optimizer = optimizer
        self.training_settings = training_settings
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.trained_model = None
        self.img_transform = img_transform

    def train(self):
        """Train the model."""
        results = trainer.train(self.training_settings, self.name)
        self.trained_model = results
        self.trained_model.set_image_transform(self.img_transform)

    def is_trained(self):
        """Check if the model has been trained."""
        return self.trained_model is not None

    def save(self):
        """
        Save the model. Will error if the model has not been trained.
        """
        if self.is_trained():
            self.trained_model.save_all()
        else:
            print("Model not yet trained. Call Model.train() first.")

    def run_on_image(self, img_path):
        """
        Run the model on an image. Will error if the model has not been trained.
        :param img_path: Path to the image to run the model on.
        :return: The output of the model.
        """
        if self.is_trained():
            return self.trained_model.run_on_image(img_path)
        else:
            raise Exception("Model not yet trained. Call Model.train() first.")

    def run_on_image_and_show(self, img_path):
        """
        Run the model on an image and return the result as a figure. Will error if the model has not been trained.
        :param img_path:
        :return:
        """
        if self.is_trained():
            return self.trained_model.run_on_image_and_show(img_path)
        else:
            raise Exception("Model not yet trained. Call Model.train() first.")

    def run_on_image_and_save_figure(self, img_path, save_filename):
        """
        Run the model on an image and save the result as a figure. Will error if the model has not been trained.
        :param img_path: Path to the image to run the model on.
        :param save_filename: Filename to save the figure as.
        Will be saved in the results/[model_name]/testing directory.
        :return: The figure
        """
        if self.is_trained():
            return self.trained_model.run_on_image_and_save_figure(img_path, save_filename)
        else:
            raise Exception("Model not yet trained. Call Model.train() first.")
