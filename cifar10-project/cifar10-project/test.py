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
train_batch_size = 50


def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    # print(np.array(images_train).shape)
    # print(np.array(labels_train).shape)
    # print (idx)
    # print(labels_train[idx,:])
    x_batch = images_train[idx, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch

cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()

# print(class_names)
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

images_train = np.array([i.flatten() for i in images_train])
images_test = np.array([i.flatten() for i in images_test])
# print("*********************")
# print(np.array(images_train).shape)
# print(np.array(labels_train).shape)
# print("*********************")
#
# print("Size of:")
# print("- Training-set:\t\t{}".format(len(images_train)))
# print("- Test-set:\t\t{}".format(len(images_test)))

# None defines that x can hold a abitrary number of image with the size img_size*img_size*num_channels
#x = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])
x = tf.placeholder(tf.float32, [None, img_size*img_size*num_channels], name="x")
# true labels associated with the images that were inputted-
y_true = tf.placeholder(tf.float32, [None, num_classes], name="labels")
W = tf.Variable(tf.zeros([img_size*img_size*num_channels, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))
y = tf.matmul(x, W) + b

#y_pred = tf.nn.softmax(tf.matmul(x, W) + b)

# Training
#y_ = tf.placeholder(tf.float32, [None, num_classes])
# print("***********************************************")
# print(x.get_shape())
# print(W.get_shape())
# print(b.get_shape())
# print(y.get_shape())
# print(y_true.get_shape())
# print("***********************************************")

cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# #loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

def weight_variable(shape, name="wv"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="bv"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# channels_out = 32
def conv_layer(x_image, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([5, 5, channels_in, channels_out], name="W"))
        b = tf.Variable(tf.truncated_normal([channels_out], name="B"))
        conv = conv2d(x_image, W)
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return max_pool_2x2(tf.nn.relu(conv + b))

def fc_layer(x_image, channels_in, channels_out, name="fc"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([8 * 8 * channels_in, channels_out],name="W"))
        b = tf.Variable(tf.truncated_normal([channels_out], name="B"))
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        return tf.nn.relu(tf.matmul(x_image, W) + b)

def Gety_cov(fc1Drop, Wfc2, Bfc2, name="fc"):
    with tf.name_scope(name):
        return tf.matmul(fc1Drop,Wfc2) + Bfc2

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

## FIRST LAYER ##
conv1 = conv_layer(x_image, num_channels, 32, "conv1")


## SECOND LAYER ##
conv2 = conv_layer(conv1, 32, 64, "conv2")


h_pool2_flat = tf.reshape(conv2, [-1, 8 * 8 * 64])

#Densely connected layer
h_fc1 = fc_layer(h_pool2_flat, 64, 1024, "fc1")

# tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout: to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
W_fc2 = tf.Variable(tf.zeros([1024, 10]))
b_fc2 = tf.Variable(tf.zeros([10]))

with tf.name_scope("fc2"):
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    tf.summary.histogram("fc2", y_conv)

###### FIRST LAYER ######
# W_conv1 = weight_variable([5, 5, num_channels, 32], name="W")
# b_conv1 = bias_variable([32], name="B")

# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)

###### SECOND LAYER ######
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

#Densely connected layer
# W_fc1 = weight_variable([8 * 8 * 64, 1024], name = "W")
# b_fc1 = bias_variable([1024], name="B")
#
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name ="fc1")

#dropout: to reduce overfitting
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# Readout Layer
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])

# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2,

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_conv))


with tf.name_scope("train"):
    adamOptimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9,
                                           beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    train_step = adamOptimizer.minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_true,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross_entropy", cross_entropy)
tf.summary.scalar("accuracy", accuracy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# writer = tf.summary.FileWriter("/Users/TorbenWVogt/Deep-learning-project/GraphData/graph")
# writer.add_graph(sess.graph)
#
# writer = tf.summary.FileWriter("/Users/TorbenWVogt/Deep-learning-project/GraphData/cleanGraph")
# writer.add_graph(sess.graph)

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/Users/TorbenWVogt/Deep-learning-project/GraphData/Summary")
writer.add_graph(sess.graph)

print("START")
for i in range(500):
    batch_xs, batch_ys = random_batch()
    if i%50 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_true: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        s = sess.run(merged_summary, feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 0.5})
        writer.add_summary(s, i)
    train_step.run(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 0.5})

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: images_test, y_true: labels_test, keep_prob: 1.0}))

# for _ in range(1000):
#   batch_xs, batch_ys = random_batch()
#   #print(batch_xs.get_shape())
#   #batch_xs = images_train[i*100:(i*100)+100]
#   #batch_ys = labels_train[i*100:(i*100)+100]
#   sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})
#
# # evaluation
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))
# #correct_prediction = tf.equal(y_pred_cls, y_true_cls)
#
#
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# # y_true = y_true.reshape(10000, img_size*img_size*num_channels)
# print(sess.run(accuracy, feed_dict={x: images_test, y_true: labels_test}))
#
#
#




