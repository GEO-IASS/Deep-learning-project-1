import numpy as np
import tensorflow as tf
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

cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = random_batch()
  sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})

# evaluation
print(tf.argmax(y,1))
print(tf.argmax(y_true,1))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#y_true = y_true.reshape(10000, img_size*img_size*num_channels)
print(sess.run(accuracy, feed_dict={x: images_test, y_true: labels_test}))







