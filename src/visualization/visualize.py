import matplotlib.pyplot as plt
import numpy as np


def visualize_metric(metric, name):
    """
    function for visualization metrics

    Visualizes given data as histogram
    :param metric: list of metrics
    :param name: name of metric
    """

    counts, bins = np.histogram(metric)
    plt.title(f'{name} scores for 1000 samples histogram')
    plt.hist(bins[:-1], weights=counts)


def visualize(train_losses, val_losses, val_bleus):
    """
    plots 3 metrics specified below

    :param train_losses: list of losses for each epoch from train
    :param val_losses: list of losses for each epoch from validation
    :param val_bleus: list of bleu scores of some sample calculated during validation

    """
    train_losses = [loss.detach().cpu() for loss in train_losses]
    val_losses = [loss.detach().cpu() for loss in val_losses]
    epochs = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(1, 3, figsize=(16, 16))
    axes[0].plot(epochs, train_losses)
    axes[1].plot(epochs, val_losses[:5])
    axes[2].plot(epochs, val_bleus[:5])
    plt.show()
