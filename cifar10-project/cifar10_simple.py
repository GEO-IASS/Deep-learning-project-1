import numpy as np
import pickle
import os

# Width and height of each image.
img_size = 32
# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3
# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels
# Number of classes.
num_classes = 10
# Number of images for each batch-file in the training-set.
_images_per_file = 10000


cifar10_path = "cifar-10-batches-py"

def convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """
    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo) #, encoding='bytes'
    return dict

def load_files():
    raw_images = []
    labels = []
    for j in range(1,6):
        batchPath = cifar10_path + '/data_batch_' + `j`
        d = unpickle(batchPath)
        raw_images.append(d['data'])
        labels.append(d['labels'])
    
    d = unpickle(cifar10_path + '/test_batch')
    raw_images.append(d['data'])
    labels = labels.append(d['labels'])
    #images = [(convert_images(raw) for raw in raw_images)]
    images = np.concatenate(raw_images)/np.float32(255)
    labels = np.concatenate(labels)
    images = np.dstack((images[:, :1024], images[:, 1024:2048], images[:, 2048:]))
    images = images.reshape((images.shape[0], 32, 32, 3)).transpose(0,3,1,2)
    
    
    return images, labels

images, labels = load_files()
print ""

