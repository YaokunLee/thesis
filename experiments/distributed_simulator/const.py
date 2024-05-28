nodenum = 2

mynodeid = 0
# 创建节点网络情况矩阵(选择使用HTTP 还是IPFS)
nodechoice = [[0] for _ in range(nodenum)]

# 将每个节点的网络情况设置为 1
for i in range(nodenum):
    nodechoice[i][0] = 1
