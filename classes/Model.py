from helpers import trainer


class Model:
    def __init__(self, name, net, hyperparams, optimizer, training_settings,
                 train_loader, test_loader, img_transform=None):
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
        results = trainer.train(self.training_settings, self.name)
        self.trained_model = results
        self.trained_model.set_image_transform(self.img_transform)

    def is_trained(self):
        return self.trained_model is not None

    def save(self):
        if self.is_trained():
            self.trained_model.save_all()
        else:
            print("Model not yet trained. Call Model.train() first.")

    def run_on_image(self, img_path):
        if self.is_trained():
            return self.trained_model.run_on_image(img_path)
        else:
            raise Exception("Model not yet trained. Call Model.train() first.")

    def run_on_image_and_show(self, img_path):
        if self.is_trained():
            return self.trained_model.run_on_image_and_show(img_path)
        else:
            raise Exception("Model not yet trained. Call Model.train() first.")

    def run_on_image_and_save_figure(self, img_path, save_filename):
        if self.is_trained():
            return self.trained_model.run_on_image_and_save_figure(img_path, save_filename)
        else:
            raise Exception("Model not yet trained. Call Model.train() first.")
