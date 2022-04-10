# llt
import time
from dataset import load_mnist
from twolayerNet import TwoLayerNet
import random
from plot_ import plot_
import pickle


def load(path):
    obj = None
    with open(path, "rb") as f:
        try:
            obj = pickle.load(f)
        except:
            print("IOError")
    return obj


start = time.time()
path = './'
train_img, train_label = load_mnist(path=path, kind='train', normalize=True)
test_img, test_label = load_mnist(path=path, kind='t10k', normalize=True)


# hyperparameter
iters_num = 50000
train_size = train_img.shape[0]
batch_size = 100
learning_rate = 0.001
random.seed(1234)

iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=500, output_size=10, reg=1e-5)


dataset = (train_img, train_label, test_img, test_label)
lr_decay = 'momentum'
training_info, best_acc, best_params = \
    network.fit(dataset, iters_num, learning_rate, lr_decay, iter_per_epoch, batch_size, loss_type='mse_error')
network.save(path + 'bestmodel')
# network = load(path + 'model')
# print(network.predict(train_img[0]))

plot_(training_info)

end = time.time()
print('Running Time: %s Seconds' %(end-start))
