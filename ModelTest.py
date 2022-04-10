# llt
from dataset import load_mnist
import numpy as np
import pickle


def load(path):
    obj = None
    with open(path, "rb") as f:
        try:
            obj = pickle.load(f)
        except:
            print("IOError")
    return obj


path = './'
train_img, train_label = load_mnist(path=path, kind='train', normalize=True)
test_img, test_label = load_mnist(path=path, kind='t10k', normalize=True)

network = load(path + 'model')
print(np.argmax(network.predict(train_img[0])), np.argmax(train_label[0]))
print(network.accuracy(test_img, test_label))


