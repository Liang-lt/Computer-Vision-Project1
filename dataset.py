# llt
import gzip
import os
import numpy as np
import struct
import matplotlib.pyplot as plt


def vectorize_label(X):
    vec = np.zeros((X.size, 10))
    for idx, row in enumerate(vec):
        row[X[idx]] = 1
    return vec


def load_mnist(path, kind='train', normalize=True):
    """
    path:数据集的路径
    kind:值为train，代表读取训练集；值为t10k，代表读取测试集
    """
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromstring(lbpath.read(), dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromstring(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
    labels = vectorize_label(labels)
    # Normalize
    if normalize:
        for img in images:
            img = img.astype(np.float32)
            img /= 255.0
    return images, labels


if __name__ == '__main__':
    path = './'
    train_images, train_labels = load_mnist(path, 'train')
    test_images, test_labels = load_mnist(path, 't10k')
    print(len(train_images))
    # print(type(train_images))

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(30):
        images = np.reshape(train_images[i], [28, 28])
        ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(images, cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(train_labels[i]))
    plt.show()
