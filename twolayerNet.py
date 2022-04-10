# llt
import numpy as np
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    # prevent from overflow
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


# 反向传播梯度
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()

        return grad


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, reg=1e-4):
        self.params = {}
        # Use Gaussian distribution to initialize
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.reg = reg

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # when label is vector
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        delta = 1e-7
        return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

    def mse_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        batch_size = y.shape[0]
        # print(y, t)
        return .5 * np.sum(np.square(y - t)) / batch_size

    def loss(self, x, t, loss_type='mse_error'):
        if loss_type == 'mse_error':
            y = self.predict(x)
            loss = self.mse_error(y, t) + 0.5 * self.reg * (np.sum(self.params['W2'] * self.params['W2'])
                                                             + np.sum(self.params['W1'] * self.params['W1']))
        else:
            y = self.predict(x)
            loss = self.cross_entropy_error(y, t) + 0.5 * self.reg * (np.sum(self.params['W2'] * self.params['W2'])
                                                                       + np.sum(self.params['W1'] * self.params['W1']))
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum( y==t ) / float(x.shape[0])
        return accuracy

    # 梯度数值解 用于测试gradient正确性
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        # print(dy.ndim, dy)
        grads['W2'] = np.dot(z1.T, dy) + 0.5 * self.reg * self.params['W2']
        grads['b2'] = np.sum(dy, axis=0) + 0.5 * self.reg * self.params['b2']
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1) + 0.5 * self.reg * self.params['W1']
        grads['b1'] = np.sum(da1, axis=0) + 0.5 * self.reg * self.params['b1']

        return grads

    def fit(self, dataset, iters_num, learning_rate, lr_decay, iter_per_epoch, batch_size, loss_type='mse_error', p=True):
        train_img, train_label, test_img, test_label = dataset
        train_size = train_img.shape[0]
        train_loss_list = []
        train_acc_list = []
        test_acc_list = []
        test_loss_list = []
        best_acc = 0
        best_params = []
        stable = 0
        max_stable = 5
        epsilon = 1e-4
        # momentum
        if lr_decay == 'momentum':
            beta = 0.05
            vw = {}
            vw['W1'] = np.zeros(self.params['W1'].shape)
            vw['W2'] = np.zeros(self.params['W2'].shape)
            vw['b1'] = np.zeros(self.params['b1'].shape)
            vw['b2'] = np.zeros(self.params['b2'].shape)
        for i in range(iters_num):
            # 获取mini-batch
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = train_img[batch_mask]
            t_batch = train_label[batch_mask]

            # grad
            # grad = network.numerical_gradient(x_batch, t_batch)
            grad = self.gradient(x_batch, t_batch)

            # update parameters
            for key in ('W1', 'b1', 'W2', 'b2'):
                if lr_decay == 'momentum':
                    vw[key] = beta * vw[key] - learning_rate * grad[key]
                    self.params[key] += vw[key]
                else:
                    self.params[key] -= learning_rate * grad[key]

            # record loss
            loss = self.loss(x_batch, t_batch, loss_type)
            train_loss_list.append(loss)

            if i % iter_per_epoch == 0:
                loss = self.loss(test_img, test_label, loss_type)
                test_loss_list.append(loss)
                train_acc = self.accuracy(train_img, train_label)
                test_acc = self.accuracy(test_img, test_label)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                if test_acc > best_acc + epsilon:
                    stable = 0
                    best_acc = test_acc
                    best_params = self.params
                else:
                    stable += 1
                    if lr_decay != 'momentum':
                        learning_rate *= 0.8
                        print('lr', learning_rate)
                    if stable > max_stable:
                        self.params = best_params
                        break

                if p is True:
                    print("iter:" + str(i) + ", train acc: " + str(train_acc) + ", test acc: " + str(test_acc))
        return (train_loss_list, train_acc_list, test_loss_list, test_acc_list), best_acc, best_params

    def save(self, path):
        obj = pickle.dumps(self)
        with open(path,"wb") as f:
            f.write(obj)


