import h5py
import os
import tensorflow as tf
from numpy.random import RandomState
import numpy as np

MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"
INPUT_NODE_NUM = 10


f = h5py.File('route_prediction.h5','r') 
data = f['data'][:]
#print(data[3])

x = tf.placeholder(tf.float32, shape = (None,INPUT_NODE_NUM), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = 'y-input')

rdm = RandomState(1)
dataset_size = 100


#真实数据
real_output = []
#为使用交叉熵损失函数,将原数据处理为1和0
for i in range(0,100):
    temp = []
    if int(data[i][62])>=0 :
        temp.append(1)
        real_output.append(temp)
    if int(data[i][62])<0 :
        temp.append(0)
        real_output.append(temp)
print("real_output:")
print(real_output)

#测试数据
TestX = rdm.rand(100,INPUT_NODE_NUM)

for i in range(0,100):
    for j in range(0,INPUT_NODE_NUM):
        TestX[i][j] = data[i][j]

#print(TestX)

w1 = tf.Variable(tf.random_normal([INPUT_NODE_NUM,10],stddev = 1))			 
w2 = tf.Variable(tf.random_normal([10,1],stddev = 1))				 
#w3 = tf.Variable(tf.random_normal([10,3],stddev = 1))				 
#w4 = tf.Variable(tf.random_normal([3,1],stddev = 1))

saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型
'''
with tf.Session() as sess:
    saver.restore(sess, "/model/model.ckpt")
    print(sess.run(w1))
'''



    
x_w1 = tf.matmul(x, w1)
y = tf.matmul(x_w1, w2)
#w2_w3 = tf.matmul(w1_w2, w3)
#y = tf.matmul(w2_w3, w4)

y = tf.sigmoid(y)
   
#测试会话开始
with tf.Session() as sess:
    #读取模型
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        pass    
    #print(sess.run(w1))
    #print(sess.run(w2))
    #print(sess.run(w3))
    '''
    for n in range(0,2):
        for i in range(0,1):
            for j in range(0,2):
                TestX[i][j] = data[i+n][j]
    '''

    predict_output = sess.run(y,{x:TestX})
    predict_outputInt = tf.round(predict_output)
    print("real_output")
    print(real_output)
    print("test_output:")
    print(predict_output)
    print(sess.run(predict_outputInt))
       
    a = real_output - predict_outputInt
    count = 0
    for i in range(0,100):
        if sess.run(a[i]) == 0:
            count = count +1
    print(count/100)
