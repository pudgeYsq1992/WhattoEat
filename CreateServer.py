from flask import Flask
from flask import request
from flask import redirect
from flask import jsonify
import json

app = Flask(__name__)
#@app.route('/api/hello', methods=['GET'])
@app.route('/', methods=['GET'])
def start(asd):
  return json.dumps({
    'code': '0'
  })

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)

'''
def hello_world():
    return 'Hello world'

if __name__ == '__main__':
    app.run()
'''