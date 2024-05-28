import threading
import time
import os
import const
import torch
# from mnist.options import args_parser

# from mnist.local_main import local_train, save_model, test_model, aggregator, sync_model

from music_recommendation.main import args_parser
from music_recommendation.main import local_train, save_model, test_model, aggregator, sync_model


# last_file_cid = {}
# for i in range(0, const.nodenum):
#     last_file_cid[i] = ""


# round 0

# 和训练相关的参数
args = args_parser(const.mynodeid, const.nodenum)
print("const.mynodeid:",const.mynodeid,"const.nodenum:",const.nodenum)
print("args:",args)

for i in range(0,1):
    # 训练阶段
    local_model = local_train(args)

    base_url = "/opt/infocomm/experiments/distributed_simulator/models"

    # 如果路径不存在的话就新建文件

    torch.save(local_model.state_dict(), base_url+"/latest_model"+str(const.mynodeid)+".pt")
            
            
    # save_model(local_model)

    # # 上传本地模型
    # if const.nodechoice[const.mynodeid][0] == 0:
    #     # 初级节点模型上传IPFS
    #     ipfsfunction.upload_file_to_ipfs_gateway(
    #         const.mynodeid, "MyModel/latest_model.pt")
    # if const.nodechoice[const.mynodeid][0] == 1:
    #     # 高级节点运行HTTP供其他节点来读取模型参数
    #     http_thread = threading.Thread(
    #         target=httpfunction.starthttp, args=("127.0.0.1", 9000+const.mynodeid))
    #     http_thread.start()

    time.sleep(5)
    local_weights = []
    # 读取其他节点模型参数
    for i in range(0, const.nodenum):
        file_list = []
        while os.path.exists(f"/opt/infocomm/experiments/distributed_simulator/models/latest_model{i+1}.pt") == False:
            print("wait")
            continue
        
        # 从本地参数文件所在文件夹中读取所有获得的参数文件
        local_model_path = "/opt/infocomm/experiments/distributed_simulator/models"
        for ff in os.listdir(local_model_path):
            file_list.append(os.path.join(local_model_path, ff))
        # 使用fedavg聚合参数
        
        for path in file_list:
            # 加载模型
            tmp_model = torch.load(path)
            # 获得其他主机的模型参数
            local_weights.append(tmp_model)
            


    # 同步来自其他节点模型
    # local_weights = sync_model(local_parameter_path = "./othersmodel")
    # 聚合模型
    local_model = aggregator(local_model, local_weights=local_weights)

    # 测试模型
    test_model(local_model)

print(f"node{const.mynodeid} finish!")
# round >0
