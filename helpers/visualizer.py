import torch
from matplotlib import pyplot as plt


def show_examples(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    for i in range(6):  # Show 6 examples
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    return fig


def show_prediction_examples(network, test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    with torch.no_grad():
        output = network(example_data)
    pred = output.data.max(1, keepdim=True)[1]
    fig = plt.figure()
    for i in range(6):  # Show 6 examples
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(pred[i].item()))
        plt.xticks([])  # Remove numbers from x-axis
        plt.yticks([])  # Remove numbers from y-axis
    return fig


def prediction_from_tensor(pred_tensor):
    pred = pred_tensor.data.max(1, keepdim=True)[1].item()
    return pred


def display_prediction(img, pred_tensor):
    pred = prediction_from_tensor(pred_tensor)
    fig = plt.figure()
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(pred))
    plt.xticks([])  # Remove numbers from x-axis
    plt.yticks([])  # Remove numbers from y-axis
    return fig
