

import os
import subprocess

import subprocess


def start_ipfs():
    try:
        ipfs_path = './ipfs_data'
        start_cmd = f'IPFS_PATH={ipfs_path} ipfs daemon'
        subprocess.Popen(start_cmd, shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("IPFS节点已在后台运行！")
    except FileNotFoundError:
        print("未找到ipfs可执行文件，请确认路径是否正确。")
    except subprocess.SubprocessError as e:
        print(f"无法启动IPFS守护进程：{e}")


def upload_file_to_ipfs_gateway(key_name, file_path):

    # 第一步：将文件添加到IPFS获取CID
    add_cmd = f'./ipfs add {file_path}'
    print(add_cmd)
    process = subprocess.Popen(
        add_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Failed to add file to IPFS: {error.decode().strip()}")
        return
    file_cid = output.decode().split(' ')[1]

    # 第三步：发布到IPNS
    publish_cmd = f' ./ipfs name publish --key={key_name} {file_cid}'
    process = subprocess.Popen(
        publish_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Failed to publish file to IPNS: {error.decode().strip()}")
        return

    print(f"Published file with CID {file_cid} to IPFS with key {key_name}")


def upload_file_to_ipfs(key_name, file_path):
    ipfs_path = './ipfs_data'

    # 第一步：将文件添加到IPFS获取CID
    add_cmd = f'IPFS_PATH={ipfs_path} ./ipfs add {file_path}'
    process = subprocess.Popen(
        add_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Failed to add file to IPFS: {error.decode().strip()}")
        return
    file_cid = output.decode().split(' ')[1]

    # 第三步：发布到IPNS
    publish_cmd = f'IPFS_PATH={ipfs_path} ./ipfs name publish --key={key_name} {file_cid}'
    process = subprocess.Popen(
        publish_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Failed to publish file to IPNS: {error.decode().strip()}")
        return

    print(f"Published file with CID {file_cid} to IPFS with key {key_name}")


def resolve_ipfs_key(key):
    # cmd = f'IPFS_PATH={ipfs_path} ./ipfs resolve -r /ipns/$(IPFS_PATH={ipfs_path} ./ipfs key list -l | grep "{key}" | awk "{{print $1}}")'
    cmd = f'./ipfs key list -l | grep -w "{key}" | awk "{{print $1}}"'

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    cid_with_node = output.decode().strip()
    # 提取CID部分
    cid = cid_with_node.split()[0]

    cmd2 = f'./ipfs resolve -r /ipns/{cid}'

    process2 = subprocess.Popen(
        cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output2, error = process2.communicate()

    return output2.decode().strip()


def resolve_ipfs_key_gateway(key):

    # cmd = f'IPFS_PATH={ipfs_path} ./ipfs resolve -r /ipns/$(IPFS_PATH={ipfs_path} ./ipfs key list -l | grep "{key}" | awk "{{print $1}}")'
    cmd = f' ./ipfs key list -l | grep -w "{key}" | awk "{{print $1}}"'

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    cid_with_node = output.decode().strip()
    print(cid_with_node)
    # 提取CID部分
    cid = cid_with_node.split()[0]

    cmd2 = f' ./ipfs resolve -r /ipns/{cid}'

    process2 = subprocess.Popen(
        cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output2, error = process2.communicate()

    return output2.decode().strip()


def ipfs_get_and_save(ipfs_hash, file_name, save_path):
    ipfs_path = './ipfs_data'
    os.makedirs(save_path, exist_ok=True)

    file_path = os.path.join(save_path, file_name)

    cmd = f'IPFS_PATH={ipfs_path} ./ipfs get {ipfs_hash} --output {file_path}'

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    print(cmd)

    if process.returncode == 0:
        print(f"File saved as {save_path}/{file_name}")
    else:
        print("Failed to get file from IPFS")


def ipfs_get_and_save_gateway(ipfs_hash, file_name, save_path):
    os.makedirs(save_path, exist_ok=True)

    file_path = os.path.join(save_path, file_name)

    cmd = f'./ipfs get {ipfs_hash} --output {file_path}'

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if process.returncode == 0:
        print(f"File saved as {save_path}/{file_name}")
    else:
        print("Failed to get file from IPFS")
