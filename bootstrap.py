import os
import shutil
import subprocess

import json

import const


def update_ipfs_config_root(config_path):
    # 打开配置文件并加载JSON数据
    with open(config_path, 'r') as config_file:
        config_data = json.load(config_file)

    # 修改 Addresses.Swarm 字段为 /ip4/127.0.0.1/tcp/8000
    config_data['Addresses']['Swarm'] = ['/ip4/127.0.0.1/tcp/10000']

    # 删除 Bootstrap 字段中的默认根节点
    config_data['Bootstrap'] = []

    # 保存修改后的配置文件
    with open(config_path, 'w') as config_file:
        json.dump(config_data, config_file, indent=2)

    print("根节点配置文件已成功更新。")


def get_node_id():
    # 执行命令获取节点 ID
    result = subprocess.run(
        ['./ipfs', 'config', 'Identity.PeerID'], capture_output=True, text=True)

    if result.returncode == 0:
        # 提取节点 ID
        node_id = result.stdout.strip()
        return node_id
    else:
        print(f"获取节点 ID 失败：{result.stderr}")
        return None


def update_ipfs_config_root(config_path):
    # 打开配置文件并加载JSON数据
    with open(config_path, 'r') as config_file:
        config_data = json.load(config_file)

    # 修改 Addresses.Swarm 字段为 /ip4/127.0.0.1/tcp/8000
    config_data['Addresses']['Swarm'] = ['/ip4/127.0.0.1/tcp/10000']

    # 删除 Bootstrap 字段中的默认根节点
    config_data['Bootstrap'] = []

    # 保存修改后的配置文件
    with open(config_path, 'w') as config_file:
        json.dump(config_data, config_file, indent=2)

    print("根节点配置文件已成功更新。")


def bootroot():
    try:
        subprocess.run(['./ipfs', 'init'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"IPFS 初始化提示：{e}")
    config_path_root = '/root/.ipfs/config'  # 根节点配置文件的路径
    update_ipfs_config_root(config_path_root)
    print("IPFS 根节点 初始化成功！")
    try:
        subprocess.Popen(['./ipfs', 'daemon'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("IPFS 根节点 守护进程已在后台运行！")
    except FileNotFoundError:
        print("未找到 ipfs 可执行文件，请确认路径是否正确。")
    except subprocess.SubprocessError as e:
        print(f"无法启动 IPFS 守护进程：{e}")
    try:
        for i in range(0, const.nodenum):
            key_name = f"{i}"
            subprocess.run(['./ipfs', 'key', 'gen', key_name], check=True)
            print(f"为节点 {i} 创建了IPNS键: {key_name}")
    except FileNotFoundError:
        print("未找到 ipfs 可执行文件，请确认路径是否正确。")


def create_node_folder(number, rootid, mynodeid):
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = f"node{number}"
    folder_path = os.path.join(current_dir, folder_name)

    # 如果文件夹已存在，则删除它
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    # 创建文件夹
    os.makedirs(folder_path)

    # 复制当前目录下的ipfs可执行文件到新创建的文件夹
    ipfs_executable = "ipfs"  # 替换为你的ipfs可执行文件名
    ipfs_executable_path = os.path.join(current_dir, ipfs_executable)
    shutil.copy2(ipfs_executable_path, folder_path)

    # 复制其他python文件到新建文件夹
    file = "const.py"
    file_path = os.path.join(current_dir, file)
    shutil.copy2(file_path, folder_path)

    file2 = "main.py"
    file_path2 = os.path.join(current_dir, file2)
    shutil.copy2(file_path2, folder_path)

    file3 = "httpfunction.py"
    file_path3 = os.path.join(current_dir, file3)
    shutil.copy2(file_path3, folder_path)

    file4 = "ipfsfunction.py"
    file_path4 = os.path.join(current_dir, file4)
    shutil.copy2(file_path4, folder_path)

    folder_name = "mnist"
    folder_path5 = os.path.join(current_dir, folder_name)
    shutil.copytree(folder_path5, os.path.join(folder_path, folder_name))

    # 在新文件夹中执行 `ipfs init` 命令并创建密钥
    ipfs_data_path = os.path.join(folder_path, "ipfs_data")
    init_command = f"IPFS_PATH={ipfs_data_path} ./{ipfs_executable} init"
    subprocess.run(init_command, shell=True)

    init_command = f"IPFS_PATH={ipfs_data_path} ./{ipfs_executable} key gen --type=rsa --size=2048 node{number}"
    subprocess.run(init_command, shell=True)

    # 修改配置文件
    config_path = os.path.join(ipfs_data_path, "config")
    with open(config_path, 'r') as config_file:
        config_data = json.load(config_file)

    # 修改本地端口为 10000+number
    config_data['Addresses']['Swarm'] = [
        f'/ip4/0.0.0.0/tcp/{10000 + number+1}']
    config_data['Addresses']['API'] = [f'/ip4/0.0.0.0/tcp/{5000 + number+2}']
    config_data['Addresses']['Gateway'] = [
        f'/ip4/0.0.0.0/tcp/{8080 + number+1}']
    # 连接根节点
    config_data['Bootstrap'] = [f'/ip4/127.0.0.1/tcp/10000/ipfs/{rootid}']
    with open(config_path, 'w') as config_file:
        json.dump(config_data, config_file, indent=2)

    # 修改 const.py 文件中的 mynodeid
    const_file_path = os.path.join(folder_path, "const.py")
    with open(const_file_path, 'r') as const_file:
        const_data = const_file.read()
    const_data = const_data.replace("mynodeid = 0", f"mynodeid = {mynodeid}")
    with open(const_file_path, 'w') as const_file:
        const_file.write(const_data)

    print(
        f"成功创建文件夹 {folder_name}，复制 ipfs 可执行文件和 const.py 文件到该文件夹内，并修改配置文件和 const.py 文件中的 mynodeid。")


bootroot()
rootid = get_node_id()
if rootid:
    print(f"当前节点的 ID 是：{rootid}")
for i in range(0, const.nodenum):
    create_node_folder(i, rootid, i)
