import h5py
import os
import tensorflow as tf
from numpy.random import RandomState
import numpy as np
import xlrd
import xlwt
import os
import operator
import string

#数据处理函数
def stringToNum(s):
    
    if s == '恶劣':
        return 0
    if s == '较差':
        return 1
    if s == '较好':
        return 2
    if s == '很好':
        return 3    
    if s == '很累':
        return 0
    if s == '有点累':
        return 1
    if s == '不大累':
        return 2
    if s == '很轻松':
        return 3
    if s == '不好吃':
        return 0 
    if s == '一般般':
        return 1
    if s == '还不错':
        return 2
    if s == '很好吃':
        return 3
    if s == '星期一':
        return 0    
    if s == '星期二':
        return 1
    if s == '星期三':
        return 2
    if s == '星期四':
        return 3
    if s == '星期五':
        return 4
    if s == '星期六':
        return 5    
    if s == '星期天':
        return 6
    if s == '便宜':
        return 0
    if s == '还行':
        return 1
    if s == '有点贵':
        return 2    
    if s == '很贵':
        return 3
    if s == '很远':
        return 0
    if s == '有些远':
        return 1
    if s == '不算远':
        return 2    
    if s == '很近':
        return 3
    if s == '一点都不想去':
        return 0
    if s == '一点也不想去':
        return 0
    if s == '不大想去':
        return 0
    if s == '随缘吧':
        return 1    
    if s == '很想去':
        return 1
    if s == '非去不可':
        return 1

MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"
INPUT_NODE_NUM = 6
workbook = xlrd.open_workbook("偏好数据/标注数据.xls")


x = tf.placeholder(tf.float32, shape = (None,INPUT_NODE_NUM), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (None, 5), name = 'y-input')

rdm = RandomState(1)
dataset_size = 100


#真实数据
real_output = []
#为使用交叉熵损失函数,将原数据处理为1和0
for i in range(0,100):
    temp = []
    temp.append(stringToNum(workbook.sheets()[0].cell(i+2,8).value))
    real_output.append(temp)
print("real_output:")
print(real_output)

#测试数据
TestX = rdm.rand(dataset_size,INPUT_NODE_NUM)

for i in range(0,100):
    TestX[i][0] = stringToNum(workbook.sheets()[0].cell(i+2,1).value)
    TestX[i][1] = stringToNum(workbook.sheets()[0].cell(i+2,2).value)
    TestX[i][2] = stringToNum(workbook.sheets()[0].cell(i+2,3).value)
    TestX[i][3] = stringToNum(workbook.sheets()[0].cell(i+2,4).value)
    TestX[i][4] = stringToNum(workbook.sheets()[0].cell(i+2,5).value)
    TestX[i][5] = stringToNum(workbook.sheets()[0].cell(i+2,6).value)

#print(TestX)

w1 = tf.Variable(tf.random_normal([INPUT_NODE_NUM,16],stddev = 1))			 
w2 = tf.Variable(tf.random_normal([16,16],stddev = 1))				 
w3 = tf.Variable(tf.random_normal([16,5],stddev = 1))				 
#w4 = tf.Variable(tf.random_normal([3,1],stddev = 1))

saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型
'''
with tf.Session() as sess:
    saver.restore(sess, "/model/model.ckpt")
    print(sess.run(w1))
'''



    
x_w1 = tf.matmul(x, w1)
x_w1 = tf.sigmoid(x_w1)
w1_w2 = tf.matmul(x_w1, w2)
w1_w2 = tf.sigmoid(w1_w2)
y = tf.matmul(w1_w2, w3)
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
    '''
    a = real_output - predict_outputInt
    count = 0
    for i in range(0,99):
        if sess.run(a[i]) <= 0.5:
            count = count +1
    print(count/100)

    rdm = RandomState(1)
    dataset_size = 1
    XX = rdm.rand(dataset_size,INPUT_NODE_NUM)
    XX[0][0] = 3
    XX[0][1] = 1
    XX[0][2] = 1
    XX[0][3] = 1
    XX[0][4] = 1
    XX[0][5] = 3
    test_output1 = sess.run(y,feed_dict ={x:XX})
    print("testoutput1:")
    print(test_output1)
    '''