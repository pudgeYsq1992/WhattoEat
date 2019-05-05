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
        return 0
    if s == '较好':
        return 1
    if s == '很好':
        return 1    
    if s == '很累':
        return 0
    if s == '有点累':
        return 0
    if s == '不大累':
        return 1
    if s == '很轻松':
        return 1
    if s == '不好吃':
        return 0 
    if s == '一般般':
        return 0
    if s == '还不错':
        return 1
    if s == '很好吃':
        return 1
    if s == '星期一':
        return 0    
    if s == '星期二':
        return 0
    if s == '星期三':
        return 0
    if s == '星期四':
        return 0
    if s == '星期五':
        return 1
    if s == '星期六':
        return 1    
    if s == '星期天':
        return 1
    if s == '便宜':
        return 0
    if s == '还行':
        return 0
    if s == '有点贵':
        return 1    
    if s == '很贵':
        return 1
    if s == '很远':
        return 0
    if s == '有些远':
        return 0
    if s == '不算远':
        return 1    
    if s == '很近':
        return 1
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

#模型存储路径及名称
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"



 
#取出标注数据
#这些数据目前存在excel表中
workbook = xlrd.open_workbook("偏好数据/标注数据.xls")



#构建 待训练模型
INPUT_NODE_NUM = 1
w1 = tf.Variable(tf.random_normal([INPUT_NODE_NUM,3],stddev = 1))			 
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1))				 
w3 = tf.Variable(tf.random_normal([3,1],stddev = 1))				 
w4 = tf.Variable(tf.random_normal([3,1],stddev = 1))

biases = tf.Variable(tf.zeros([2]))
biases2 = tf.Variable(tf.zeros([3]))
#模型的输入输出
x = tf.placeholder(tf.float32, shape = (None,INPUT_NODE_NUM), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = 'y-input')

#使用随机数填满数组
rdm = RandomState(1)
dataset_size = 100
X = rdm.rand(dataset_size,INPUT_NODE_NUM)

#读取数据写入数组
#X为训练数据
for i in range(0,100):
    X[i][0] = i
    '''
    if (i%2 == 0):
        X[i][0] = 1
    else:
        X[i][0] = 1
    '''
        
    '''
    X[i][0] = stringToNum(workbook.sheets()[0].cell(i+2,1).value)
    X[i][1] = stringToNum(workbook.sheets()[0].cell(i+2,2).value)
    X[i][2] = stringToNum(workbook.sheets()[0].cell(i+2,3).value)
    X[i][3] = stringToNum(workbook.sheets()[0].cell(i+2,4).value)
    X[i][4] = stringToNum(workbook.sheets()[0].cell(i+2,5).value)
    X[i][5] = stringToNum(workbook.sheets()[0].cell(i+2,6).value)
    '''
print(X)
#真实数据
Y = []
for i in range(0,100):
    '''
    temp = []
    if X[i][0] >0:
        temp.append(0)
        Y.append(temp)
    else :
        temp.append(0)
        Y.append(temp)
    '''
    temp = []
    temp.append(i)
    Y.append(temp)
print(Y)

'''
for i in range(0,100):
    temp = []
    temp.append(stringToNum(workbook.sheets()[0].cell(i+2,8).value))
    Y.append(temp)
print(Y)
'''
x_w1 = tf.matmul(x, w1) 
#x_w1 = tf.nn.relu(x_w1)
#x_w1 = tf.sigmoid(x_w1)

y = tf.matmul(x_w1, w2)
#w1_w2 = tf.nn.relu(w1_w2) 
#w1_w2 = tf.sigmoid(w1_w2)

#y = tf.matmul(w1_w2, w3)
#w2_w3 = tf.nn.relu(w2_w3)
#w2_w3 = tf.sigmoid(w2_w3)
'''
y = tf.matmul(w2_w3, w4)
#y = tf.nn.relu(y)
'''
y = tf.sigmoid(y)

writer = tf.summary.FileWriter("/path/to/log",tf.get_default_graph())
writer.close()

cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    +(1-y)*tf.log(tf.clip_by_value(1-y, 1e-10,1.0)))
'''
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y)
'''
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)


batch_size = 2


saver = tf.train.Saver()

#训练会话开始
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    #初始化变量
    sess.run(init_op)
    STEPS = 100000
 
    for i in range(STEPS): 
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y[start:end]})

        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict = {x:X,y_:Y})
            print("After %d training step(s) ,cross entropy on all data is %g"%(i,total_cross_entropy))
            #print("After %d training step(s)"%(i))
    print("w1:")
    print(sess.run(w1))
    print("w2:")
    print(sess.run(w2))
    print("w3:")
    print(sess.run(w3))
    
    saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME))
    test_output = sess.run(y,feed_dict ={x:X})
    #print("X[0:1] :")
    #print(X[0:1])
    
    #inferenced_y = np.argmax(test_output,1)
    
    print("testoutput:")
    print(test_output)
    
    
