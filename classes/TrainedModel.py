import os

import torch
from PIL import Image
from matplotlib import pyplot as plt

from helpers import visualizer


class TrainedModel:
    """A model that has been trained and tested"""
    def __init__(self, name, model, test_loader, train_losses, train_counter, test_losses, test_counter,
                 save_dir=os.getcwd() + "/results/", img_transform=None):
        """
        Creates a new TrainedModel
        :param name: The name of the model
        :param model: The trained model, including weights
        :param test_loader: The test loader used to test the model
        :param train_losses: The losses during training
        :param train_counter: The number of training examples seen during training
        :param test_losses: The losses during testing
        :param test_counter: The number of training examples seen during testing
        :param save_dir: The directory to save the results to
        :param img_transform: The image transform used to transform the images
        """
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
        """
        Creates a plot of the training and testing losses
        :return: The plot
        """
        fig = plt.figure()
        plt.plot(self.train_counter, self.train_losses, color='blue')
        plt.scatter(self.test_counter, self.test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('negative log likelihood loss')
        return fig

    def to_text(self):
        """
        Converts the model to a string
        :return:
        """
        return "Train losses: " + str(self.train_losses) + "\n" + \
               "Train counter: " + str(self.train_counter) + "\n" + \
               "Test losses: " + str(self.test_losses) + "\n" + \
               "Test counter: " + str(self.test_counter) + "\n"

    def save_to_file(self, filename):
        """
        Converts the model to a string and saves it to a file
        :param filename: The filename to save to
        """

        filename = filename + ".txt"
        print("Saving training results to " + filename)
        with open(filename, "w") as f:
            f.write(self.to_text())

    def save_plot_to_file(self, filename):
        """
        Generates a plot and saves it to a file.
        :param filename: The filename to save to
        :return:
        """
        filename = filename + ".png"
        print("Saving training plot to " + filename)
        self.make_plot().savefig(filename)

    def save_weights(self, filename):
        """
        Saves the model weights to a file
        :param filename: The filename to save to
        :return:
        """
        filename = filename + ".pth"
        print("Saving weights to " + filename)
        torch.save(self.model.state_dict(), filename)

    def show_prediction_examples(self):
        """
        Shows a figure with prediction examples
        :return: The figure
        """
        return visualizer.show_prediction_examples(self.model, self.test_loader)

    def save_prediction_examples(self, filename):
        """
        Saves the prediction examples to a file
        :param filename: The filename to save to
        :return:
        """
        filename = filename + ".png"
        print("Saving prediction examples to " + filename)
        self.show_prediction_examples().savefig(filename)

    def init_directories(self):
        """
        Initializes the directories for the results
        :return:
        """
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
        """
        Saves all training results to a file
        :return:
        """
        self.init_directories()
        filepath = self.root_directory_path + "/training"
        self.save_plot_to_file(filepath + "/training_plot")
        self.save_weights(filepath + "/model_weights")
        self.save_to_file(filepath + "/training_results")

    def save_all_testing(self):
        """
        Saves all testing results to a file
        :return:
        """
        self.init_directories()
        filepath = self.root_directory_path + "/testing"
        self.save_prediction_examples(filepath + "/prediction_examples")

    def save_all(self):
        """
        Saves all results to a file
        :return:
        """
        print("Saving all results to " + self.root_directory_path)
        self.save_all_training()
        self.save_all_testing()

    def set_image_transform(self, transform):
        """
        Sets the image transform
        :param transform:  The image transform
        :return:
        """
        self.img_transform = transform

    def run_on_image(self, img_path):
        """
        Runs the model on an image
        :param img_path: The path to the image
        :return:
        """
        if self.img_transform is None:
            raise Exception("No image transform set. "
                            "Please set one using TrainedModel.set_image_transform([transform])")
        img = Image.open("data/single-images/" + img_path)
        image_transformed = self.img_transform(img)
        return self.model(image_transformed)

    def run_on_image_and_make_figure(self, img_path):
        """
        Runs the model on an image and creates a figure with the result
        :param img_path: The path to the image
        :return:
        """
        img = Image.open("data/single-images/" + img_path)
        result = self.run_on_image(img_path)
        return visualizer.display_prediction(img, result)

    def run_on_image_and_save_figure(self, img_path, save_filename):
        """
        Runs the model on an image and saves the result to a file
        :param img_path: The path to the image
        :param save_filename: The filename to save to
        :return:
        """
        filepath = self.root_directory_path + "/testing"
        save_filename = save_filename + ".png"
        filepath = filepath + "/" + save_filename
        print("Saving prediction for " + img_path + " to " + filepath)
        self.run_on_image_and_make_figure(img_path).savefig(filepath)



