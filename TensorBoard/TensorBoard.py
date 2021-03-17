import tensorflow as tf

with tf.name_scope("Input"):
    x_ = tf.compat.v1.placeholder(tf.float32, shape = [4,2], name = "x-input-predicates")
    tf.summary.image("input predicates x", x_, 10)

with tf.name_scope("Input_expected_output"):
    y_ = tf.compat.v1.placeholder(tf.float32, shape = [4,1], name = "y-expected-output")
    tf.summary.image("input expected values of y", y_, 10)


X_NOR = [[0,0],[0,1],[1,0],[1,1]]
W1 = tf.Variable(tf.random_uniform_initializer([2,2], -1, 1), name="Weights1")
B1 = tf.Variable(tf.zeros([2]), name="Bias1")
LS = tf.sigmoid(tf.matmul(X_NOR,W1) + B1)
output = tf.sigmoid(tf.matmul(LS,))