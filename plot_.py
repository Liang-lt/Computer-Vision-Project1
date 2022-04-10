# llt
import matplotlib.pyplot as plt
import numpy as np


def plot_(info):
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = info
    plt.figure(figsize=(20, 10))
    # Loss
    x1 = np.arange(len(train_loss_list))
    x2 = np.arange(0, len(train_loss_list), len(train_loss_list) / len(test_loss_list))
    ax1 = plt.subplot(211)
    plt.plot(x1, train_loss_list)
    plt.plot(x2, test_loss_list, 'ro-')
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend(['Training Loss', 'Testing Loss'])

    # Accuracy
    markers = {'train': 'o', 'test': 's'}
    x2 = np.arange(len(train_acc_list))
    ax2 = plt.subplot(212)
    plt.plot(x2, train_acc_list, label='train acc')
    plt.plot(x2, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
