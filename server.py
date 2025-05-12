from flask import Flask, jsonify, request
from solution import AlgSolution
import logging
import os
import sys

agent = AlgSolution()
app = Flask(__name__)

app.config['MAX_FORM_MEMORY_SIZE'] = 64 * 1024 * 1024


@app.route('/step', methods=['POST'])
def step():
    form = request.form
    action = agent.predicts(**form)
    return jsonify(**action)

@app.route('/reset', methods=['POST'])
def reset():
    form = request.form
    agent.reset(**form)
    return jsonify(message="success")

@app.route('/synchronize', methods=['GET'])
def synchroize():
    return jsonify(message="success")

@app.route('/stop', methods=['POST'])
def stop():
    msg = request.form['msg']
    f = open('/home/admin/workspace/job/logs/user.log', 'a')
    f.write(msg)
    f.close()
    return jsonify(message="success")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
