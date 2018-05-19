import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def fc_mnist():
    print("Done loading MNIST")
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # tensor of size anything x 10. get the one hot?
    y_true = tf.placeholder(tf.float32, [None, 10])
    # -sum y' log(y)
    # y' = truth, y = pred?
    # since ytrue is 0/1s one hot, we'll take the log of everything
    # from the predict but when we multiply, thats element wise and we
    # only get y_true * log(y_pred of that truth)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})

        if i % 100 == 0:
            print("Iter {} done".format(i))


    # evaluate
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_mnist():
    # stuff to feed into the dictionary
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_true = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # variables
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    w_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    w_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    # densly connected layer
    w_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    # first conv layer
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second conv layer
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # dropout prior to readout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                                           logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = mnist.train.next_batch(64)
        if i % 100 == 0 and i > 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_true: batch[1], keep_prob:1.0})
            print("Step {} | training accuracy: {}".format(i, train_accuracy))

        train_step.run(feed_dict={x:batch[0], y_true: batch[1], keep_prob:0.5})

    # can do tf.Variable.eval(feeddict )
    print("Test accuracy {}".format(accuracy.eval(feed_dict={x: batch[0], y_true: batch[1], keep_prob:1.0})))


conv_mnist()
