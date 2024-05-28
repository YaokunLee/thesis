# 去中心化联邦学习框架代码使用说明



### 0. 实验数据以及依赖软件配置：

- 实验数据集link： https://www.dropbox.com/sh/j12um64gsig5wqk/AAD4Vov6hUGwbLoVxh3wASg_a?dl=0，存放在Nowplaying数据集下
- IPFS kubo的安装：https://docs.ipfs.tech/install/command-line/#system-requirements
- 依赖软件：
  - python == 3.9
  - numpy  ==1.22.4
  - fedlab ==1.1.2
  - torch == 2.0.1 




**实验1:证明移动网络中异质性和动态性对DFL的影响:** 

**1.1 异质性**

- 通过***createfile.py*** 来创建固定大小的临时文件用于传输。(文章中的实验是20MB)

##### 图4：测量IPFS所占用的Memory

- 搭建一个十节点的IPFS网络，可以通过***bootstrap.py*** 来进行自动生成，若生成出现错误，则按照以下步骤进行手动生成。
  - 参考：https://blog.csdn.net/Lyon_Nee/article/details/112990326。
  - node1 启动:IPFS_PATH=node1/ ipfs daemon &
  - 查询node1 id: IPFS_PATH=./node1/ ipfs id 
  - node2 将node1 添加为引导节点: IPFS_PATH=./node2/ ipfs bootstrap add /ip4/127.0.0.1/tcp/14001/ipfs/ node1 id
  - node2 启动: IPFS_PATH=./node2/ ipfs daemon
  - node 3-10 同理将node1 添加为引导节点后启动

- 通过 ***checkmemory.py*** 来检测进程占用的Memory,会输出一个memory_usage.csv文档。注意：如果需要同时检测多个进程，请修改代码中文件存储的名称来避免覆盖。
- 运行 ***IPFS_upload_download.py*** 模拟节点进行上传和下载文件，若模拟出现错误，则也可以通过节点的操作命令进行进行手动上传下载。



##### 图5：测量HTTP占用的Memory

- 通过 ***httpserver.py*** 和 ***httpclient.py*** 来实现 文件通过HTTP进行传输。
- 同样通过 ***checkmemory.py*** 来检测进程占用的Memory,会输出一个memory_usage.csv文档。注意：如果需要同时检测多个进程，请修改代码中文件存储的名称来避免覆盖。



**1.2 动态性**

动态性模型单机模拟节点上下线:

1.进入experiment/music目录

2.运行converage_experiment_main.py文件，结果输出的路径可以在变量res_file中设置

3.输出结果



**实验2: 评估我们提出框架（MobileDFL）的性能**

- 运行 ***bootstrap.py***  可以生成对应数量的node文件夹，代表着不同的文件。
- 每个node文件夹下运行main文件。注意，每个epoch生成的模型命名为 ***MyModel/latest_model.pt***
- Memory的检测使用和上面一致的 ***checkmemory.py*** ；storage 的检测 可以直接查看node下model文件保存的数据大小；download data 的检测通过端口接受的流量。



**分布式模型训练执行步骤：**

1. 进入子目录 distributed_simulator/

2. 设置const.py中nodenum(参与节点的数量) 参数

3. 运行bootstrap.py生成参与的节点

4. 各节点运行main.py 进行训练，此次main.py中是示例，可根据目标任务不同进行代码更改。



