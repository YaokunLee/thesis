import threading
import time
import ipfsfunction
import httpfunction
import const

from music_recommendation.main import args_parser
from music_recommendation.main import local_train, save_model, test_model, aggregator, sync_model


last_file_cid = {}
for i in range(0, const.nodenum):
    last_file_cid[i] = ""


num_epochs = 100



# 和训练相关的参数
args = args_parser(const.mynodeid, const.nodenum)

# 判断是否为高级节点
if const.nodechoice[const.mynodeid][0] == 1:
    # 启动IPFS
    ipfsfunction.start_ipfs()
    print("本节点为高级节点，已经启动IPFS")

for epoch in range(num_epochs):
    # 训练阶段
    local_model = local_train(args)
    save_model(local_model)

    # 上传本地模型
    if const.nodechoice[const.mynodeid][0] == 0:
        # 初级节点模型上传IPFS
        ipfsfunction.upload_file_to_ipfs_gateway(
            const.mynodeid, "MyModel/latest_model.pt")
    if const.nodechoice[const.mynodeid][0] == 1:
        # 高级节点运行HTTP供其他节点来读取模型参数
        http_thread = threading.Thread(
            target=httpfunction.starthttp, args=("127.0.0.1", 9000+const.mynodeid))
        http_thread.start()

    time.sleep(5)
    # 读取其他节点模型参数
    for i in range(0, const.nodenum):
        if i != const.mynodeid:
            # check 节点网络情况矩阵
            if const.nodechoice[i][0] == 1:
                nowcid = httpfunction.get_cid_from_ip('127.0.0.1', 9000+i)
                if nowcid != last_file_cid[i]:
                    httpfunction.get_file_from_ip(
                        '127.0.0.1', 9000+i, f'./othersmodel/node{i}.pt')
                    print(f"通过http读取了node{i}的参数")

            elif const.nodechoice[i][0] == 0:
                if const.nodechoice[const.mynodeid][0] == 0:
                    nowcid = ipfsfunction.resolve_ipfs_key_gateway(i)
                    if nowcid != last_file_cid[i]:
                        ipfsfunction.ipfs_get_and_save_gateway(
                            nowcid, f"node{i}.pt", './othersmodel')
                        print(f"通过IPFS读取了node{i}的参数")
                    else:
                        print(f"node{i},当前未产生新模型参数(与上一epoch相同)")
                elif const.nodechoice[const.mynodeid][0] == 1:
                    nowcid = ipfsfunction.resolve_ipfs_key(i)
                    if nowcid != last_file_cid[i]:
                        ipfsfunction.ipfs_get_and_save_gateway(
                            nowcid, f"node{i}.pt", './othersmodel')
                        print(f"通过IPFS读取了node{i}的参数")
                    else:
                        print(f"node{i},当前未产生新模型参数(与上一epoch相同)")


    # 同步来自其他节点模型
    local_weights = sync_model(local_parameter_path = "./othersmodel")
    # 聚合模型
    local_model = aggregator(local_model, local_weights=local_weights)

    # 测试模型
    test_model(local_model)
    print(f"node{i},结束了第{epoch}轮")


