import h5py
import os
import tensorflow as tf
from numpy.random import RandomState
import numpy as np

MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"
INPUT_NODE_NUM = 10

f = h5py.File('route_prediction.h5','r') 
#取出数据,存在data中,data是一个数组
data = f['data'][:]
print(data)

		
w1 = tf.Variable(tf.random_normal([INPUT_NODE_NUM,10],stddev = 1))			 
w2 = tf.Variable(tf.random_normal([10,1],stddev = 1))				 
w3 = tf.Variable(tf.random_normal([10,3],stddev = 1))				 
w4 = tf.Variable(tf.random_normal([3,1],stddev = 1))

x = tf.placeholder(tf.float32, shape = (None,INPUT_NODE_NUM), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = 'y-input')

rdm = RandomState(1)
dataset_size = 100

X = rdm.rand(100,INPUT_NODE_NUM)

for i in range(0,100):
    for j in range(0,INPUT_NODE_NUM):
        X[i][j] = data[i][j]
print(X)


#真实数据
Y = []
#为使用交叉熵损失函数,将原数据处理为1和0
for i in range(0,100):
    temp = []
    if int(data[i][62])>=0 :
        temp.append(0)
        Y.append(temp)
    if int(data[i][62])<0 :
        temp.append(1)
        Y.append(temp)

print(Y)


'''
#模拟数据
Y = []
#Y = [[int(x1+x2 < 1)] for (x1,x2) in X]
for i in range(0,100):
    temp = []
    if X[i][0] >=0:
        temp.append(0)
        Y.append(temp)
    else :
        temp.append(0)
        Y.append(temp)
print(Y)
'''

x_w1 = tf.matmul(x, w1)
y = tf.matmul(x_w1, w2)
#w2_w3 = tf.matmul(w1_w2, w3)
#y = tf.matmul(w2_w3, w4)

y = tf.sigmoid(y)

cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    +(1-y)*tf.log(tf.clip_by_value(1-y, 1e-10,1.0)))

#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y)
#cross_entropy = tf.reduce_mean(y_-y)


train_step = tf.train.AdamOptimizer(0.000001).minimize(cross_entropy)

batch_size = 1

#训练过程可视化
#writer = tf.summary.FileWriter("/path/to/log",tf.get_default_graph)
#writer.close()

saver = tf.train.Saver()

#训练会话开始
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    #初始化变量
    sess.run(init_op)
    STEPS = 500000
    for i in range(STEPS): 
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict = {x:X,y_:Y})
            print("After %d training step(s) ,cross entropy on all data is %g"%(i,total_cross_entropy))
    print("w1:")
    print(sess.run(w1))
    print("w2:")
    print(sess.run(w2))
    print("w3:")
    print(sess.run(w3))
    saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME))
    #saver.save(sess, "module/model.ckpt")
    test_output = sess.run(y,{x:X})
    inferenced_y = np.argmax(test_output,1)
    print("testoutput:")
    print(test_output)
    
writer = tf.summary.FileWriter("/path/to/log",tf.get_default_graph())
writer.close()