import requests
import os
import hashlib
from flask import Flask, send_file

app = Flask(__name__)

# 开启 Flask 应用程序


def starthttp(ip, port):
    app.run(host=ip, port=port)

# 获取 model 文件的 cid


def get_file_cid():
    file_path = './MyModel/latest_model.pt'
    with open(file_path, 'rb') as file:
        content = file.read()
        hash_object = hashlib.sha256(content)
        cid = hash_object.hexdigest()
    return cid

# 处理 /filecid 路由


@app.route('/filecid')
def file_cid():
    cid = get_file_cid()
    return cid

# 处理 /file 路由


@app.route('/file')
def get_file():
    filename = './MyModel/latest_model.pt'  # 替换为实际的文件路径
    return send_file(filename)


def get_cid_from_ip(ip, port):
    url = f'http://{ip}:{port}/filecid'
    response = requests.get(url)
    cid = response.text
    return cid


def get_file_from_ip(ip, port, save_path):
    url = f'http://{ip}:{port}/file'
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as file:
            file.write(response.content)
        return True
    else:
        return False
