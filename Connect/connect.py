import os
import shutil


def generate_ipfs_config(new_ip, new_port):
    # 创建新的IPFS配置文件目录和数据目录
    config_dir = os.path.expanduser(f'~/.ipfs_new_{new_port}')
    data_dir = os.path.expanduser(f'~/.ipfs_data_{new_port}')
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 生成新的IPFS配置文件路径
    new_config_file = os.path.join(config_dir, 'config')

    # 生成新的IPFS配置文件内容
    config_data = f"""
    [Addresses]
        Swarm = ["/ip4/0.0.0.0/tcp/{new_port}"]
        API = "/ip4/127.0.0.1/tcp/{new_port+1}"
        Gateway = "/ip4/127.0.0.1/tcp/{new_port+2}"

    [Bootstrap]
        Addresses = []
    """

    # 写入新的IPFS配置文件
    with open(new_config_file, 'w') as f:
        f.write(config_data)

    return config_dir, data_dir


def start_ipfs_node(config_dir, data_dir):
    # 设置环境变量指向新的IPFS配置文件目录和数据目录
    os.environ['IPFS_PATH'] = config_dir
    os.environ['IPFS_MONITORING'] = 'true'
    os.environ['IPFS_MONITORING_CONFIG'] = f'{config_dir}/monitoring.json'
    os.environ['IPFS_MONITORING_PUBLISHED_HOST'] = 'localhost'
    os.environ['IPFS_MONITORING_PUBLISHED_PORT'] = '4001'

    # 启动新的IPFS节点
    os.system(
        f'ipfs init --profile=server --empty-repo --bits 4096 --datastore={data_dir}')
    os.system('ipfs daemon')


# 主循环
while True:
    # 获取新的IP和端口输入
    new_ip = input("请输入新的IP地址：")
    new_port = int(input("请输入新的端口号："))

    # 生成新的IPFS配置文件和数据目录
    config_dir, data_dir = generate_ipfs_config(new_ip, new_port)

    # 启动新的IPFS节点并连接到本地已开启的节点
    start_ipfs_node(config_dir, data_dir)
