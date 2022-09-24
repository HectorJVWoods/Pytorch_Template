import os

import torch
from PIL import Image
from matplotlib import pyplot as plt

from helpers import visualizer


class TrainedModel:
    def __init__(self, name, model, test_loader, train_losses, train_counter, test_losses, test_counter,
                 save_dir=os.getcwd() + "/results/", img_transform=None):
        self.name = name
        self.model = model
        self.model.eval()
        self.train_losses = train_losses
        self.train_counter = train_counter
        self.test_losses = test_losses
        self.test_counter = test_counter
        self.save_dir = save_dir
        self.test_loader = test_loader
        self.root_directory_path = save_dir + self.name + "/"
        self.img_transform = img_transform

    def make_plot(self):
        fig = plt.figure()
        plt.plot(self.train_counter, self.train_losses, color='blue')
        plt.scatter(self.test_counter, self.test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('negative log likelihood loss')
        return fig

    def to_text(self):
        return "Train losses: " + str(self.train_losses) + "\n" + \
               "Train counter: " + str(self.train_counter) + "\n" + \
               "Test losses: " + str(self.test_losses) + "\n" + \
               "Test counter: " + str(self.test_counter) + "\n"

    def save_to_file(self, filename):
        filename = filename + ".txt"
        print("Saving training results to " + filename)
        with open(filename, "w") as f:
            f.write(self.to_text())

    def save_plot_to_file(self, filename):
        filename = filename + ".png"
        print("Saving training plot to " + filename)
        self.make_plot().savefig(filename)

    def save_weights(self, filename):
        filename = filename + ".pth"
        print("Saving weights to " + filename)
        torch.save(self.model.state_dict(), filename)

    def show_prediction_examples(self):
        return visualizer.show_prediction_examples(self.model, self.test_loader)

    def save_prediction_examples(self, filename):
        filename = filename + ".png"
        print("Saving prediction examples to " + filename)
        self.show_prediction_examples().savefig(filename)

    def init_directories(self):
        main_dir = self.root_directory_path
        if not os.path.exists(main_dir):
            print("Creating missing directory " + main_dir)
            os.makedirs(main_dir)
        training_dir = main_dir + "/training"
        if not os.path.exists(training_dir):
            print("Creating missing subdirectory " + training_dir)
            os.makedirs(training_dir)
        testing_dir = main_dir + "/testing"
        if not os.path.exists(testing_dir):
            print("Creating missing subdirectory " + testing_dir)
            os.makedirs(testing_dir)

    def save_all_training(self):
        self.init_directories()
        filepath = self.root_directory_path + "/training"
        self.save_plot_to_file(filepath + "/training_plot")
        self.save_weights(filepath + "/model_weights")
        self.save_to_file(filepath + "/training_results")

    def save_all_testing(self):
        self.init_directories()
        filepath = self.root_directory_path + "/testing"
        self.save_prediction_examples(filepath + "/prediction_examples")

    def save_all(self):
        print("Saving all results to " + self.root_directory_path)
        self.save_all_training()
        self.save_all_testing()

    def set_image_transform(self, transform):
        self.img_transform = transform

    def run_on_image(self, img_path):
        if self.img_transform is None:
            raise Exception("No image transform set. "
                            "Please set one using TrainedModel.set_image_transform([transform])")
        img = Image.open("data/single-images/" + img_path)
        image_transformed = self.img_transform(img)
        return self.model(image_transformed)

    def run_on_image_and_make_figure(self, img_path):
        img = Image.open("data/single-images/" + img_path)
        result = self.run_on_image(img_path)
        return visualizer.display_prediction(img, result)

    def run_on_image_and_save_figure(self, img_path, save_filename):
        filepath = self.root_directory_path + "/testing"
        save_filename = save_filename + ".png"
        filepath = filepath + "/" + save_filename
        print("Saving prediction for " + img_path + " to " + filepath)
        self.run_on_image_and_make_figure(img_path).savefig(filepath)



