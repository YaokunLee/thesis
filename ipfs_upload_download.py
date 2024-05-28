import string
import subprocess
import random
import time
import concurrent.futures


def create_unique_file(file_path):
    data = ''.join(random.choices(string.ascii_letters +
                   string.digits, k=200*1024*1024))  # 生成 10MB 的随机字符串
    with open(file_path, 'w') as file:
        file.write(data)


def upload_file_to_ipfs(file_path, ipfs_path):
    upload_command = f'IPFS_PATH={ipfs_path} ./ipfs add -Q --raw-leaves --chunker=size-262144 {file_path}'
    ipfs_process = subprocess.run(
        upload_command, shell=True, capture_output=True, text=True)
    if ipfs_process.returncode == 0:
        ipfs_hash = ipfs_process.stdout.strip()
        print(f"文件已上传至 IPFS 网络，IPFS 哈希为：{ipfs_hash}", ipfs_path)
        return ipfs_hash
    else:
        print("文件上传失败。")
        return None


def download_file_from_ipfs(ipfs_hash, ipfs_path):
    download_command = f'IPFS_PATH={ipfs_path} ./ipfs get -o downloaded_file.txt {ipfs_hash}'
    ipfs_process = subprocess.run(
        download_command, shell=True, capture_output=True, text=True)
    if ipfs_process.returncode == 0:
        print("文件已从 IPFS 网络下载成功。", ipfs_path)
    else:
        print("文件下载失败。")


if __name__ == '__main__':
    ipfs_paths = [f"node{i}/ipfs_data" for i in range(10)]  # 节点的执行路径列表
    file_path = '20MB_file.txt'
    epoch = 0

    while True:
        # 创建随机的 20MB 大小文件
        create_unique_file(file_path)
        epoch = epoch+1
        print(epoch)

        # 随机选择一个节点上传文件
        upload_ipfs_path = random.choice(ipfs_paths)
        ipfs_hash = upload_file_to_ipfs(file_path, upload_ipfs_path)

        # 并发下载文件
        with concurrent.futures.ThreadPoolExecutor() as executor:
            download_tasks = []
            for download_ipfs_path in ipfs_paths:
                if download_ipfs_path != upload_ipfs_path:
                    # 提交下载任务到线程池
                    download_task = executor.submit(
                        download_file_from_ipfs, ipfs_hash, download_ipfs_path)
                    download_tasks.append(download_task)

            # 等待所有下载任务完成
            for download_task in concurrent.futures.as_completed(download_tasks):
                try:
                    download_task.result()
                except Exception as e:
                    print(f"下载任务出现异常: {str(e)}")

        # time.sleep(1)  # 等待 1 秒后进行下一次循环
