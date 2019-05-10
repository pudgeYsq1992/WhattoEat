#coding=utf-8
from flask import Flask
from flask import request
from flask import redirect
from flask import jsonify
import json

'''
def test():  
    return json.dumps({
        'Second stage':'start'
    })
'''

app = Flask(__name__)
#@app.route('/api/hello', methods=['GET'])
@app.route('/', methods=['GET'])
def start():
    return json.dumps({
        'Well come to Yans party': 'Hello DingMingjie and ChenCheng'
    })


@app.route('/<strD>',methods=['GET'])
def create_app(strD):
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
    return json.dumps({
        'hello':'guy'
    })


@app.route('/signin' , methods=['GET'])
def signin():
    # 需要从request对象读取表单内容：
    if request.form['username'] == 'admin' and request.form['password'] == 'password':
        return '<h3>Hello, admin!</h3>'
    return '<h3>Bad username or password.</h3>'

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)

