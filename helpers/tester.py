import torch
import torch.nn.functional as func


def test(network, test_loader, test_losses=None):
    """
    Test the network on the test set.
    :param network: The network to test.
    :param test_loader: The test set.
    :param test_losses: The list of test losses (from previous steps). If None, a new list will be created.
    :return:
    """
    if test_losses is None:
        test_losses = []
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += func.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    total = len(test_loader.dataset)
    percentage_correct = correct / total
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total, accuracy))
    test_losses.append(test_loss)
    return test_losses, accuracy, correct, total, percentage_correct
