import os
import shutil
import subprocess
import multiprocessing

def create_node_folder(number):
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = f"node{number}"
    folder_path = os.path.join(current_dir, folder_name)

    # 如果文件夹已存在，则删除它
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    # 创建文件夹
    os.makedirs(folder_path)
    
    # 复制其他python文件到新建文件夹
    file = "const.py"
    file_path = os.path.join(current_dir, file)
    shutil.copy2(file_path, folder_path)
    
    
    file = "main.py"
    file_path = os.path.join(current_dir, file)
    shutil.copy2(file_path, folder_path)
    
    # 复制其他文件夹到新建文件夹

    folder_name = "music_recommendation"
    folder_path5 = os.path.join(current_dir, folder_name)
    shutil.copytree(folder_path5, os.path.join(folder_path, folder_name))
    
     # 修改 const.py 文件中的 mynodeid
    const_file_path = os.path.join(folder_path, "const.py")
    with open(const_file_path, 'r') as const_file:
        const_data = const_file.read()
    const_data = const_data.replace("mynodeid = 0", f"mynodeid = {number}")
    with open(const_file_path, 'w') as const_file:
        const_file.write(const_data)

def start_train(path):
    # for i in range(0,10):
    
    p = subprocess.getoutput(["python3.9",path])
    # 等待子进程完成
    # output, error = p.communicate()

    # 获取子进程的返回值
    # return_code = p.returncode

    # 将标准输出转换为字符串
    # output_str = output.decode('utf-8')
    # print("return_code:",return_code)
    # print("output_str:",output_str)
    
    print("finished")
    return p

for i in range(0,2):
    create_node_folder(i+1)


# for i in range(0,1):
#     for i in range(0,2):
#         os.system(f"cd node{i+1}")
        
#         p = start_train(path = f"node{i+1}/main.py")
#         print("p:",p)
