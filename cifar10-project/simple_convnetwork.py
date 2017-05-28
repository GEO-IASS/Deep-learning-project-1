import numpy as np
import tensorflow as tf
import cifar10
from cifar10 import img_size, num_channels, num_classes

img_size_cropped = 24
train_batch_size = 50

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    x_batch = images_train[idx, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch

cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()

print(class_names)
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

images_train = np.array([i.flatten() for i in images_train])
images_test = np.array([i.flatten() for i in images_test])

# None defines that x can hold a abitrary number of image with the size img_size*img_size*num_channels
#x = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])
x = tf.placeholder(tf.float32, [None, img_size*img_size*num_channels])
# true labels associated with the images that were inputted-
y_true = tf.placeholder(tf.float32, [None, num_classes])
W = tf.Variable(tf.zeros([img_size*img_size*num_channels, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))
y = tf.matmul(x, W) + b

W_conv1 = weight_variable([5, 5, num_channels, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([8 * 8 * 64, 1024*3])
b_fc1 = bias_variable([1024*3])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([img_size*img_size*3, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Evaluation functions
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(5000):
  batch = random_batch()
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_true: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: images_test, y_true: labels_test, keep_prob: 1.0}))
