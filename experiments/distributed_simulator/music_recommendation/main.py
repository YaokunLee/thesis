import argparse
import pickle
from .model import train_alone,test_alone
import time
from .util import Data, split_validation
from .model import *
import os
from munch import Munch
from copy import deepcopy
from fedlab.utils.serialization import SerializationTool
from .trainer import SubsetSerialTrainer
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.dataset import SubsetSampler
import copy
import warnings

warnings.filterwarnings("ignore")
import random
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  
 
seed_torch()

import os

# 获取当前脚本所在目录的绝对路径
script_dir = os.path.dirname(__file__)

# 数据集文件夹的相对路径
dataset_dir = "Nowplaying/"

# 数据集文件夹的绝对路径
abs_dataset_dir = os.path.join(script_dir, dataset_dir)

def args_parser(nodeid, nodenum):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Nowplaying', help='dataset name: diginetica/Nowplaying/sample')
    parser.add_argument('--epoch', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--embSize', type=int, default=100, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
    parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
    parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')


    # 联邦相关参数
    # parser.add_argument("--client_num", default=10, type=int, help="客户机总数")
    parser.add_argument('--idx', type=int, default=nodeid,
                        help="client index for dataset distribution")
    parser.add_argument('--num_users', type=int, default=nodenum,
                        help="客户机总数")
    parser.add_argument('--load_model', type=bool, default=False,
                        help="是否加载模型")
    parser.add_argument('--load_path', type=str, default=".MyModel/latest_model.pt",
                        help="加载模型的位置")
    parser.add_argument("--aggregate_model", type=bool, default=True,
                        help="本轮是否需要做聚合任务")
    # parser.add_argument("--client_num", default=10, type=int, help="客户机总数")
    parser.add_argument("--sample_ratio", default=1, type=float, help="取多少客户的参数做聚合")
    parser.add_argument("--com_round", default=1, type=int, help="运行几轮")
    
    return parser.parse_args()




# 分割数据集
def _get_subdataset(client_id, train_data_size, test_data_size, client_size, train_dataset, test_dataset):
    scale_factor =0.2
    # 按照客户机数量均分数据集
    train_piece_size = int(scale_factor*train_data_size//client_size)
    test_piece_size = int(scale_factor*test_data_size//client_size)
    # print("train_piece_size:",train_piece_size)
    # print("test_piece_size",test_piece_size)
    train_start_index =  train_piece_size * (client_id - 1)
    train_end_index = train_piece_size * (client_id - 1) + train_piece_size
    print("train_start_index:", train_start_index)
    print("train_end_index:",train_end_index)
    test_start_index =  test_piece_size * (client_id - 1)
    test_end_index = test_piece_size * (client_id - 1) + test_piece_size
    
    
    train_data=[train_dataset[0][train_start_index:train_end_index],train_dataset[1][train_start_index:train_end_index]]
   
    test_data = [test_dataset[0][test_start_index:test_end_index],test_dataset[1][test_start_index:test_end_index]]

    return train_data, test_data

# 加载数据集
def load_dataset():
    train_data = pickle.load(open(abs_dataset_dir + '/train.txt', 'rb'))
    test_data = pickle.load(open(abs_dataset_dir + '/test.txt', 'rb'))
    # print("train data len:",len(train_data[0]),len(train_data[1]))
    # print("test data len:",len(test_data[0]),len(test_data[1]))
    return train_data, test_data

# 加载模型
def load_model(model,file_name):
    original_state_dict = model.state_dict()
    # print(original_state_dict.keys())
    new_dict = torch.load(file_name)
    # print(new_dict.keys())
    for key in new_dict:
        original_state_dict[key]=new_dict[key]
    model.load_state_dict(original_state_dict)
    return model


def save_model(global_model):
    # 保存当前模型
    base_url = "MyModel/"

    # 如果路径不存在的话就新建文件
    if os.path.exists("MyModel") != True:
        os.mkdir("MyModel")
        torch.save(global_model.state_dict(), base_url+"latest_model.pt")
    else:
        torch.save(global_model.state_dict(), base_url+"latest_model.pt")
        
        
# 测试模型
def test_model(client_model, test_data_total, total_loss):
    
    top_K = [5, 10, 20]
    metrics = test_alone(client_model, test_data =test_data_total)
    
    best_results = {}
    
    for K in top_K:
        # best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for K in top_K:
        metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
        metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
        if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
            best_results['metric%d' % K][0] = metrics['hit%d' % K]
            # best_results['epoch%d' % K][0] = epoch
        if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
            best_results['metric%d' % K][1] = metrics['mrr%d' % K]
            # best_results['epoch%d' % K][1] = epoch


    for K in top_K:
        print('train_loss:\t%.5f\tHIT@%d: %.5f\tMRR%d: %.5f\t' %
            (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1]))
        
    return best_results


def aggregator(global_model, local_weights):
    # 如果需要聚合就使用聚合后的模型进行test
    aggregator = Aggregators.fedavg_aggregate
    serialize_model_list = []
    # 把同步到的模型进行序列化
    for w in local_weights:
        serialize_model_list.append(SerializationTool.serialize_model(w))
    # 加入本地模型序列化后的结果
    serialize_model_list.append(SerializationTool.serialize_model(global_model))
    # update global weights
    global_weights = aggregator(serialize_model_list)
    # 将聚合后的模型参数加载到本地模型中
    global_model.load_state_dict(SerializationTool.deserialize_model(global_weights))

    return global_model

def sync_model(local_parameter_path="./othersmodel"):
    '''
    Args:
         local_parameter_path:本地存放参数文件的地址

    Return:
         返回参数列表
    '''

    # 从本地参数文件所在文件夹中读取所有获得的参数文件
    file_list = []
    for ff in os.listdir(local_parameter_path):
        file_list.append(os.path.join(local_parameter_path, ff))
    # 使用fedavg聚合参数
    param_list = []
    for path in file_list:
        # 加载模型
        tmp_model = torch.load(path)
        # 获得其他主机的模型参数
        param_list.append(tmp_model)
    return param_list


#  训练主函数
def local_train(opt):
    print(opt)
    res_file = open("a.txt",'a+')
    total_train_data,total_test_data = load_dataset()
    

    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'Nowplaying':
        n_node = 60416
    else:
        n_node = 309

    # 包含所有数据的集合
    train_data_total = Data(total_train_data, shuffle=False, n_node=n_node)
    test_data_total = Data(total_test_data, shuffle=True, n_node=n_node)
    
    print("train data len:",len(total_train_data[0]),len(total_train_data[1]))
    print("test data len:",len(total_test_data[0]),len(total_test_data[1]))
    # 初始化客户机模型
    client_model = trans_to_cpu(DHCN(adjacency=train_data_total.adjacency,n_node=n_node,lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset))
    
    # 加载本地模型
    if opt.load_model:
        client_model = load_model(client_model, opt.load_path)
    
    
    print("opt.idx:",opt.idx)
    # 获取当前用户训练所需的数据
    client_train_data, client_test_data = _get_subdataset(client_id=opt.idx, train_data_size=len(total_train_data[0]), test_data_size = len(total_test_data[0]),client_size=opt.num_users, 
                                                            train_dataset=total_train_data, test_dataset = total_test_data)
    print("client_train_data:",len(client_train_data[0]))
    client_train_data = Data(client_train_data, shuffle=False, n_node=n_node)
    client_test_data = Data(client_test_data, shuffle=False, n_node=n_node)
    
    
    # 开始训练当前用户的模型   
    print(
        "Starting training procedure of client [{}]".format(opt.idx))
    print('-------------------------------------------------------')
   
    for epoch in range(opt.epoch):
        res_file.write("start round:"+str(epoch)+'\n')
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        client_model, total_loss= train_alone(client_model, train_data=client_train_data)
    
    best_results = test_model(client_model=client_model,test_data_total = test_data_total, total_loss = total_loss)
    print("best_results:",best_results)
    return client_model

# opt = args_parser(1,10)
# local_train(opt)