import numpy as np
import tensorflow as tf
import cifar10
from cifar10 import img_size, num_channels, num_classes

# Hyperparameters
num_epochs=1
l2_regularization_penalty = 0.01
learning_rate=0.001
decay_rate_1_moment=0.9
decay_rate_2_moment=0.999
epsilon=1e-8
conv_strides = [1, 1, 1, 1]
pooling_strides = [1, 2, 2, 1]
window_size = [1, 2, 2, 1]
use_batch_norm = False

# training batch size
train_batch_size = 50
is_training = True

# Function for weight creation, initialized with small noise
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Function for bias creation, initialized with small value because we use relu
# to avoid dead neurons
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# function for creating convolution layer, with specified stride value
# and using padding
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=conv_strides, padding='SAME')

# Max pool with 2x2 block
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=window_size,
                        strides=pooling_strides, padding='SAME')

def random_batch():
    num_images = len(images_train)
    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)
    # Extract batch                           
    x_batch = images_train[idx, :]
    y_batch = labels_train[idx, :]
    return x_batch, y_batch

#Load image data - cifar10
cifar10.maybe_download_and_extract()
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

#Print class names
class_names = cifar10.load_class_names()
print(class_names)

#Create image as one hot vector
images_train = np.array([i.flatten() for i in images_train])
images_test = np.array([i.flatten() for i in images_test])

# None defines that x can hold a abitrary number of image with the size img_size*img_size*num_channels
# x = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])
x = tf.placeholder(tf.float32, [None, img_size*img_size*num_channels])
# true labels associated with the images that were inputted-
y_true = tf.placeholder(tf.float32, [None, num_classes])


# First layer - compute 32 features for each 3 x3 patch
# [ 3, 3, num_channels, 32] is [3x3, input_channels, output_cannels]

W_conv1 = weight_variable([3, 3, num_channels, 32])
b_conv1 = bias_variable([32])
tf.summary.histogram("W1", W_conv1)
tf.summary.histogram("b1", b_conv1)

# Reshape image to a 4d tensor before applying 1st convolution layer
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

# Apply convolution -> Relu -> Max pooling
conv1 = conv2d(x_image, W_conv1) + b_conv1
if(use_batch_norm):
  conv1 = tf.layers.batch_normalization(conv1, epsilon=epsilon, training=is_training)
h_conv1 = tf.nn.relu(conv1)

tf.summary.histogram("activation1", h_conv1)

with tf.name_scope("conv1"):
    h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer - 3x3 patch
# [3, 3, 32, 64] 64 features for each 3x3 patch

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
tf.summary.histogram("W2", W_conv2)
tf.summary.histogram("b2", b_conv2)

# Apply convolutional -> Relu -> Max pooling
h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
if(use_batch_norm):
  h_conv2 = tf.layers.batch_normalization(h_conv2, epsilon=epsilon, training=is_training)
h_conv2 = tf.nn.relu(h_conv2)
tf.summary.histogram("activation2", h_conv2)

with tf.name_scope("conv2"):
    h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer with 1024 neurons
# After pooling the images is reduced to 8 x 8, with 64 features
W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])

# Apply fc layer -> relu 
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

with tf.name_scope("fc1"):
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    tf.summary.histogram("fc1", h_fc1)

# Apply dropout before readout layer
# keep_prop placeholder allows to turn it on and off during training
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

with tf.name_scope("fc2"):
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    tf.summary.histogram("fc2", y_conv)
# Loss function - using softmax and L2 regularization

    unregularized_loss  = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_conv))

l2_loss = l2_regularization_penalty * 0.5 * (tf.nn.l2_loss(W_conv1) + 
                                       tf.nn.l2_loss(W_conv2) +
                                       tf.nn.l2_loss(W_fc1))
with tf.name_scope("loss"):
    loss = tf.add(unregularized_loss, l2_loss, name='loss')

with tf.name_scope("train"):
    # Choosing optimizer
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=decay_rate_1_moment, beta2=decay_rate_2_moment, epsilon=epsilon, use_locking=False).minimize(loss)

with tf.name_scope("accuracy"):
    #Accuacy
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_true,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Tensorboard

tf.summary.scalar("cross_entropy", loss)
training_acc = tf.summary.scalar("training_accuracy", accuracy)
test_acc = tf.summary.scalar("test_accuracy", accuracy)


# Start training
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/Users/TorbenWVogt/Deep-learning-project/GraphData/Summary")
writer.add_graph(sess.graph)

for i in range(int((len(images_train)/train_batch_size) * num_epochs)):
  batch = random_batch()
  if i%100 == 0:
    if i % (int(len(images_train)/train_batch_size)) == 0:   
      print("%d th epoch"%(i/(len(images_train)/train_batch_size)))
      
    # train_accuracy = train_accuracy.eval(feed_dict={x:batch[0], y_true: batch[1], keep_prob: 1.0})
    # print("step %d, training accuracy %g"%(i, train_accuracy))
    print (i)

    # To log training accuracy.
    train_acc, train_summ = sess.run(
        [accuracy, training_acc],
        feed_dict={x : batch[0], y_true : batch[1], keep_prob: 1.0})
    writer.add_summary(train_summ, i)

    # To log validation accuracy.
    valid_acc, test_summ = sess.run(
        [accuracy, merged_summary],
        feed_dict={x : images_test, y_true : labels_test, keep_prob: 1.0})
    writer.add_summary(test_summ, i)

  train_step.run(feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.5})

is_training = False
#Final validation accuracy
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: images_test, y_true: labels_test, keep_prob: 1.0}))
