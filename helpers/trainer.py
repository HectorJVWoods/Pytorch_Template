import torch
import torch.nn.functional as func

# TODO: add support for training without loaders
from classes.TrainedModel import TrainedModel
from helpers.tester import test


def train_one_epoch(train_loader, network, optimizer, epoch, train_losses, train_counter,
                    hyperparameters):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = func.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % hyperparameters.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
    return train_losses, train_counter


def load_trained_model(model_name, net, test_loader, img_transform=None):
    model_filename = "results/" + model_name + "/training/" + "model_weights.pth"
    results_filename = "results/" + model_name + "/training/" + "training_results.txt"
    model_state_dict = torch.load(model_filename)
    net.load_state_dict(model_state_dict)
    with open(results_filename, "r") as f:
        lines = f.readlines()
        train_losses = eval(lines[0].split(": ")[1])
        train_counter = eval(lines[1].split(": ")[1])
        test_losses = eval(lines[2].split(": ")[1])
        test_counter = eval(lines[3].split(": ")[1])
    trained_model = TrainedModel(model_name, net, test_loader, train_losses, train_counter, test_losses, test_counter)
    trained_model.set_image_transform(img_transform)
    return trained_model


def train(training_settings, saved_model_name="autosave"):
    network = training_settings.network
    test_loader = training_settings.test_loader
    train_loader = training_settings.train_loader
    hyperparams = training_settings.hyperparams
    optimizer = training_settings.optimizer
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(training_settings.train_loader.dataset) for i in range(hyperparams.n_epochs + 1)]
    test_losses, _, _, _, _ = test(network, test_loader, test_losses)
    for epoch in range(1, hyperparams.n_epochs + 1):
        train_losses, train_counter = train_one_epoch(train_loader, network, optimizer,
                                                      epoch, train_losses, train_counter, hyperparams)
        test_losses, _, _, _, _ = test(network, test_loader, test_losses)
    return TrainedModel(saved_model_name, network, test_loader, train_losses, train_counter, test_losses, test_counter)
