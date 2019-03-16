# encoding:utf-8
import os
import numpy as np
import tensorflow as tf
from collections import Counter
import tensorflow.contrib.keras as kr

class Text(object):
    # 打开文件
    def open_file(self,filename,mode='r'):
        return open(filename,mode,encoding='utf-8',errors='ignore')

    # 读取文件
    def read_file(self,filename):
        print('读取文件...')
        contents,labels = [],[]
        with self.open_file(filename) as f:
            f = f.readlines()[1:]
            for line in f:
                try:
                    label,content = line.strip().split('\t')
                    if content:
                        contents.append(list(content))
                        labels.append(label)
                except:
                    pass
        return contents,labels

    # 读取词汇表，一个词对应一个id
    def read_vocab(self,vocab_dir):
        with self.open_file(vocab_dir) as fp:
            words = [_.strip() for _ in fp.readlines()]
        word_to_id = dict(zip(words,range(len(words))))
        return words,word_to_id

    # 读取分类目录，一个类别对应一个id
    def read_category(self):
        categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
        cat_to_id = dict(zip(categories, range(len(categories))))
        return categories, cat_to_id

    # 根据训练集构建词汇表，存储
    def build_vocab(self,train_dir,vocab_dir,vocab_size=5000):
        data_train,_ = self.read_file(train_dir)
        all_data = []
        for content in data_train:
            all_data.extend(content)
        counter = Counter(all_data)
        count_pairs = counter.most_common(vocab_size - 1)
        words,_ = list(zip(*count_pairs))
        # 添加一个<PAD>来将所有文本pad为同一个长度
        words = ['<PAD>'] + list(words)
        self.open_file(vocab_dir,mode='w').write('\n'.join(words) + '\n')

    # 将文件转换为id表示
    def process_file(self,filename,word_to_id,cat_to_id,max_length=600):
        contents,labels = self.read_file(filename)

        data_id,label_id = [],[]
        for i in range(len(contents)):
            data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            label_id.append(cat_to_id[labels[i]])

        # 使用keras提供的pad_sequences来将文本转为固定长度，不足的补0
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id,max_length)
        y_pad = kr.utils.to_categorical(label_id,num_classes=len(cat_to_id)) # 将标签转换为one-hot表示
        return x_pad,y_pad

    # 获取数据
    def get_data(self,filename,text_length):
        vocab_dir = 'C:/Users/lejon/YuanPrograms/DataWhile_NLP/cnews/cnews.vocab.txt'
        categories,cat_to_id = self.read_category()
        words,word_to_id = self.read_vocab(vocab_dir)
        x,y = self.process_file(filename,word_to_id,cat_to_id,text_length)
        return x,y

class TextCNN(object):
    def __init__(self):
        self.text_length = 600 # 文本长度
        self.num_classer = 10 # 类别数

        self.vocab_size = 5000 # 词汇表大小
        self.word_vec_dim = 64 # 词向量维度

        self.filter_width = 5 # 卷积核尺寸
        self.filter_width_list = [2,3,4,5] # 卷积核尺寸列表
        self.num_filters = 10 # 卷积核数目

        self.dropout_prob = 0.5 # dropout
        self.learning_rate = 0.005 # 学习率
        self.iter_num = 10 # 迭代次数
        self.batch_size = 64 # 每轮迭代训练的样本量
        self.model_save_path = './model/' # 模型保存路径
        self.model_name = 'mnist_model'  # 模型的命名
        self.embedding = tf.get_variable('embedding',[self.vocab_size,self.word_vec_dim])

        self.fc1_size = 32 # 第一层全连接的神经元个数
        self.fc2_size = 64 # 第二层全连接的神经元个数
        self.fc3_size = 10 # 第三层全连接的神经元个数

    # 定义初始化网络权重函数
    def get_weight(self,shape,regularizer):
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1)) # 生成hshape的正太分布数据
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)) # 为权重加入L2正则化
        return w

    # 定义初始化偏置函数
    def get_bias(self,shape):
        b = tf.Variable(tf.ones(shape))
        return b

    # 模型1，使用多种卷积核
    def model_1(self,x,is_train):
        # embedding层
        embedding_res = tf.nn.embedding_lookup(self.embedding,x) # 寻找x中对应的self.embedding矩阵行数据。

        pool_list = []
        for filter_width in self.filter_width_list:
            # 卷积层
            conv_w = self.get_weight([filter_width,self.word_vec_dim,self.num_filters],0.01)
            conv_b = self.get_bias([self.num_filters])
            conv = tf.nn.conv1d(embedding_res,conv_w,stride=1,padding='VALID')
            conv_res = tf.nn.relu(tf.nn.bias_add(conv,conv_b))

            # 最大池化
            pool_list.append(tf.reduce_max(conv_res,reduction_indices=[1]))
        pool_res = tf.concat(pool_list,1)

        # 第一个全连接
        fc1_w = self.get_weight([self.num_filters*len(self.filter_width_list),self.fc1_size],0.01)
        fc1_b = self.get_bias([self.fc1_size])
        fc1_res = tf.nn.relu(tf.matmul(pool_res,fc1_w)+fc1_b)
        if is_train:
            fc1_res = tf.nn.dropout(fc1_res,0.5)

        # 第二个全连接层
        fc2_w = self.get_weight([self.fc1_size,self.fc2_size],0.01)
        fc2_b = self.get_bias([self.fc2_size])
        fc2_res = tf.nn.relu(tf.matmul(fc1_res,fc2_w)+fc2_b)
        if is_train:
            fc2_res = tf.nn.dropout(fc2_res,0.5)

        # 第三个全连接层
        fc3_w = self.get_weight([self.fc2_size,self.fc3_size],0.01)
        fc3_b = self.get_bias([self.fc3_size])
        fc3_res = tf.matmul(fc2_res,fc3_w)+fc3_b
        return fc3_res

    # 生成批次数据
    def batch_iter(self,x,y):
        data_len = len(x)
        num_batch = int((data_len-1)/self.batch_size)+ 1
        indices = np.random.permutation(np.arange(data_len)) # 随机打乱一个数组
        x_shuffle = x[indices]
        y_shuffle = y[indices]
        for i in range(num_batch):
            start = i*self.batch_size
            end = min((i+1)*self.batch_size,data_len)
            yield x_shuffle[start:end],y_shuffle[start:end]

# 训练
def train(TextCNN,trainx,trainy):
    print('model训练....')
    x = tf.placeholder(tf.int32,[None,TextCNN.text_length])
    y = tf.placeholder(tf.float32,[None,TextCNN.num_classer])
    y_pred = TextCNN.model_1(x,True)

    # 声明一个全局计数器，并输出为0，存放到目前为止模型迭代的次数
    global_step = tf.Variable(0,trainable=False)

    # 损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y)
    loss = tf.reduce_mean(cross_entropy)

    # 优化器
    train_step = tf.train.AdamOptimizer(learning_rate=TextCNN.learning_rate).minimize(loss,global_step=global_step)

    saver = tf.train.Saver() # 实例化一个保存和恢复变量的saver

    # 创建一个会话
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 通过checkpoint文件定位到最新保存的模型
        ckpt = tf.train.get_checkpoint_state(TextCNN.model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            # 加载最新模型
            saver.restore(sess,ckpt.model_checkpoint_path)

        # 循环迭代，每次迭代读取一个batch_size大小的数据
        for i in range(TextCNN.iter_num):
            batch_train = TextCNN.batch_iter(trainx,trainy)
            for x_batch,y_batch in batch_train:
                loss_value,step = sess.run([loss,train_step],feed_dict={x:x_batch,y:y_batch})
                print('After %d training step(s), loss on training batch is %g.' % (i, loss_value))
                saver.save(sess,os.path.join(TextCNN.model_save_path,TextCNN.model_name),global_step=global_step)

# 预测
def predict(TextCNN,testx,testy):
    print('model预测....')
    # 创建一个默认图，在该图中执行以下操作
    x = tf.placeholder(tf.int32,[None,TextCNN.text_length])
    y = tf.placeholder(tf.float32,[None,TextCNN.num_classer])
    y_pred = TextCNN.model1(x,False)

    saver = tf.train.Saver()

    correct_prediction = tf.equal(tf.arggmax(y,1),tf.argmax(y_pred,1))# 判断预测值和实际值是否相同
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求平均得到准确率

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(TextCNN.model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)

            # 根据读入的模型名字切分出该模型是属于迭代了多少次保存的
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split(' ')[-1]

            # 计算出测试集上的准确率
            accuracy_score = sess.run(accuracy,feed_dict={x:testx,y:testy})
            print('After %s training step(s), test accuracy = %g' % (global_step, accuracy_score))
        else:
            print('No checkpoint file found')
            return

if __name__ == '__main__':
    text_length = 600
    text = Text()
    path = 'C:/Users/lejon/YuanPrograms/DataWhile_NLP/cnews/'
    X_train,y_train = text.get_data(path+'cnews.train.txt',text_length)
    X_test, y_test = text.get_data(path+'cnews.test.txt', text_length)
    X_val, y_val = text.get_data(path+'cnews.val.txt', text_length)

    is_train = True
    textcnn = TextCNN()
    if is_train:
        train(textcnn,X_train,y_train)
    else:
        predict(textcnn,X_val,y_val)





