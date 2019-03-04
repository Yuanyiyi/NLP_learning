# encoding:utf-8

# 计算机可视化
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

# 卷积层,固定部分参数
def conv_layer(input,channels_in,channels_out):
    # channels_in: 输入通道
    # channels_out: 输出通道
    weights = tf.Variable(tf.truncated_normal([5,5,channels_in,channels_out],stddev=0.05))
    biases = tf.Variable(tf.constant(0.05,shape=[channels_out]))
    conv = tf.nn.conv2d(input,filter=weights,strides=[1,1,1,1],padding='SAME')
    act = tf.nn.relu(conv+biases)
    return act

# 简化全连接层
def fc_layer(input,num_inputs,num_outputs,use_relu=True):
    weights = tf.Variable(tf.truncated_normal([num_inputs,num_outputs],stddev=0.05))
    biases = tf.Variable(tf.constant(0.05,shape=[num_outputs]))
    act = tf.matmul(input,weights)+biases
    if use_relu:
        act = tf.nn.relu(act)
    return act

# max pooling 层
def max_pool(input):
    return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 载入数据，构建网络
path = 'C:/Users/lejon/tensorLearn/MNIST_data'
data = input_data.read_data_sets(path,one_hot=True)
x = tf.placeholder(tf.float32,shape=[None,784]) # 固定此部分值
y = tf.placeholder(tf.float32,shape=[None,10])
x_image = tf.reshape(x,[-1,28,28,1])

conv1 = conv_layer(x_image,1,32) # 增加了卷积核数目
pool1 = max_pool(conv1)

conv2 = conv_layer(pool1,32,64)
pool2 = max_pool(conv2)

flat_shape = pool2.get_shape()[1:4].num_elements()
flattened = tf.reshape(pool2,[-1,flat_shape])

fc1 = fc_layer(flattened,flat_shape,1024) # 增大神经元数目
logits = fc_layer(fc1,1024,10,use_relu=False)

# 交叉熵，优化器，准确率
# 计算交叉熵
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))
# 使用Adam优化器来训练
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
# 计算准确率
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y,axis=1),tf.argmax(logits,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 创建session，训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_batch_size = 100
    for i in range(200):
        x_batch,y_batch = data.train.next_batch(train_batch_size)
        feed_dict = {x:x_batch,y:y_batch}
        if i%100==0:
            train_accuracy = sess.run(accuracy,feed_dict=feed_dict)
            print('迭代次数：{0:>6},训练准确率:{1:>6.4%}'.format(i,train_accuracy))
        sess.run(optimizer,feed_dict=feed_dict)
    tensorboard_dir = path+'tensorboard/mnist'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(sess.graph)

