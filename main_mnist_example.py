import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from classes.Model import Model

# Reference main.py file for classifying the MNIST dataset.
# You can use this file to create your own models for whatever dataset you want.
# Credit to https://nextjournal.com/gkoehler/pytorch-mnist for the model architecture and hyperparameters
from classes.TrainingHyperparameters import TrainingHyperparameters
from classes.TrainingSettings import TrainingSettings
from helpers.trainer import load_trained_model

img_transform = torchvision.transforms.Compose([  # Define a transform to convert images to tensors
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
])
train_loader = torch.utils.data.DataLoader(  # Load training data from torchvision.datasets and apply the transform
    torchvision.datasets.MNIST('data/', train=True, download=True,
                               transform=img_transform),
    batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(  # Load test data from torchvision.datasets and apply the transform
    torchvision.datasets.MNIST('data/', train=False, download=True,
                               transform=img_transform),
    batch_size=1000, shuffle=True)


# Define Network Architecture to classify MNIST images. You can use any architecture you want, so long as it extends
# nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = func.relu(func.max_pool2d(self.conv1(x), 2))
        x = func.relu(func.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = func.relu(self.fc1(x))
        x = func.dropout(x, training=self.training)
        x = self.fc2(x)
        return func.log_softmax(x, dim=1)


# Initialise the network
network = Net()
# Define hyperparameters for training
hyperparams = TrainingHyperparameters(16, 64, 1000, 0.01, 0.5, 10)
# Define optimizer. In this case we will use stochastic gradient descent
optimizer = optim.SGD(network.parameters(), lr=hyperparams.learning_rate, momentum=hyperparams.momentum)
# Create a TrainingSettings object to pass to the trainer
training_settings = TrainingSettings(network, hyperparams, optimizer, train_loader, test_loader)
# Train the network. Also tests the network after each epoch.
model = Model("MNIST-16-epochs", network, hyperparams, optimizer, training_settings,
              train_loader, test_loader, img_transform)
model.train()
# Save the trained model (as well as test results) to results/sixteen-epochs/
model.save()
# Test the network on a single image, save the result to sixteen-epochs/testing
# You can call this method on any image in data/single-images
model.run_on_image_and_save_figure("six.png", "six-result")
# Use this method to load a trained model from disk. You must provide the test_loader and img_transform.
# TrainedModel can be tested and saved in the same way as Model, but cannot be retrained.
trained_model = load_trained_model("MNIST-16-epochs", network, test_loader, img_transform)
trained_model.run_on_image_and_save_figure("six.png", "six-result")


