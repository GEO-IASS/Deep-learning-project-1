# from data_utils import load_CIFAR10

import numpy as np
import tensorflow as tf


# img_height = 32
# img_width = 32
# channels = 3
# img_size = 1024
# num_of_classes = 10
# img_size_flat = img_height * img_width * channels
# # CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
# # Load the raw CIFAR-10 data
# cifar10_dir = 'cifar-10-batches-py'
# X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

import cifar10
from cifar10 import img_size, num_channels, num_classes

img_size_cropped = 24
train_batch_size = 64


def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    print(np.array(images_train).shape)
    print(np.array(labels_train).shape)
    print (idx)
    print(labels_train[idx,:])
    x_batch = images_train[idx, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch

cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()

print(class_names)
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

images_train = [i.flatten() for i in images_train]
images_test = [i.flatten() for i in images_train]
print("*********************")
print(np.array(images_train).shape)
print(np.array(labels_train).shape)
print("*********************")

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

# None defines that x can hold a abitrary number of image with the size img_size*img_size*num_channels
#x = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])
x = tf.placeholder(tf.float32, [None, img_size*img_size*num_channels])
# true labels associated with the images that were inputted-
y_true = tf.placeholder(tf.float32, [None, num_classes])
W = tf.Variable(tf.zeros([img_size*img_size*num_channels, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))
y = tf.matmul(x, W) + b

#y_pred = tf.nn.softmax(tf.matmul(x, W) + b)

# Training
#y_ = tf.placeholder(tf.float32, [None, num_classes])
print("***********************************************")
print(x.get_shape())
print(W.get_shape())
print(b.get_shape())
print(y.get_shape())
print(y_true.get_shape())
print("***********************************************")

cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# #loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = random_batch()
  #print(batch_xs.get_shape())
  #batch_xs = images_train[i*100:(i*100)+100]
  #batch_ys = labels_train[i*100:(i*100)+100]
  sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})

# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))
#correct_prediction = tf.equal(y_pred_cls, y_true_cls)


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_true = y_true.reshape(10000, img_size*img_size*num_channels)
print(sess.run(accuracy, feed_dict={x: images_test, y_true: labels_test}))







