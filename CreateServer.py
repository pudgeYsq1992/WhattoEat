#coding=utf-8
from flask import Flask
from flask import request
from flask import redirect
from flask import jsonify
import json

#模型部分
import os
import tensorflow as tf
from numpy.random import RandomState
import numpy as np

MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"
INPUT_NODE_NUM = 6

x = tf.placeholder(tf.float32, shape = (None,INPUT_NODE_NUM), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (None, 5), name = 'y-input')
w1 = tf.Variable(tf.random_normal([INPUT_NODE_NUM,16],stddev = 1))			 
w2 = tf.Variable(tf.random_normal([16,16],stddev = 1))				 
w3 = tf.Variable(tf.random_normal([16,5],stddev = 1))				 
#w4 = tf.Variable(tf.random_normal([3,1],stddev = 1))
saver = tf.train.Saver() 
x_w1 = tf.matmul(x, w1)
x_w1 = tf.sigmoid(x_w1)
w1_w2 = tf.matmul(x_w1, w2)
w1_w2 = tf.sigmoid(w1_w2)
y = tf.matmul(w1_w2, w3)
y = tf.sigmoid(y)

def outputByChinese(outputArray,sess):
    if sess.run(predict_outputInt)[0][0] == 1:
        resultStr = "你恐怕是一点也不想去这家店。。。"
    if sess.run(predict_outputInt)[0][1] == 1:
        resultStr = "你应该不大想去吧"
    if sess.run(predict_outputInt)[0][2] == 1:
        resultStr = "多半是随缘了，有人拉你也就去了"
    if sess.run(predict_outputInt)[0][3] == 1:
        resultStr = "你应该很想去这家店吧"
    if sess.run(predict_outputInt)[0][4] == 1:
        resultStr = "你非去不可，拦都拦不住"

app = Flask(__name__)
#@app.route('/api/hello', methods=['GET'])
@app.route('/', methods=['GET'])
def start():
    return json.dumps({
        'Well come to Yans party': 'Hello DingMingjie and ChenCheng'
    })


@app.route('/<strD>',methods=['GET'])
def create_app(strD):
    InputX = rdm.rand(1,INPUT_NODE_NUM)
    with tf.Session() as sess:
    #读取模型
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            pass    
        InputX[0][0] = strD[0]
        InputX[0][1] = strD[1]
        InputX[0][2] = strD[2]
        InputX[0][3] = strD[3]
        InputX[0][4] = strD[4]
        InputX[0][5] = strD[5]   
        predict_output = sess.run(y,{x:InputX})
        predict_outputInt = tf.round(predict_output)
        output_string = outputByChinese(predict_outputInt,sess)
    '''
    if strD == 'CC':
        return json.dumps({
            'Nice try': 'Hello,CC'
        })
    if strD == 'huhao':
        return json.dumps({
            'hello':'xiao biao fu'
        })
    if strD == 'Dingmingjie':
        return json.dumps({
            'Nice try':'Dingding'
        })
    if strD == '000000':
        return json.dumps({
            'This is good':'hello'
        })
    '''
    return json.dumps({
        'hello': output_string
    })


@app.route('/signin' , methods=['GET'])
def signin():
    # 需要从request对象读取表单内容：
    if request.form['username'] == 'admin' and request.form['password'] == 'password':
        return '<h3>Hello, admin!</h3>'
    return '<h3>Bad username or password.</h3>'

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)

