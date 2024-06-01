import numpy as np


def load_data(filename, num_images):
    with open(filename, 'rb') as f:
        f.read(16)
        buf = f.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28, 28)
    return data


def load_labels(filename, num_labels):

    with open(filename, 'rb') as f:
        f.read(8)
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


if __name__ == "__main__":
    train_data = load_data('./data/train-images-idx3-ubyte', 60000)
    train_labels = load_labels('./data/train-labels-idx1-ubyte', 60000)

    test_data = load_data('./data/t10k-images-idx3-ubyte', 10000)
    test_labels = load_labels('./data/t10k-labels-idx1-ubyte', 10000)