
import argparse
import pickle
import time
from util import Data, split_validation
from model import *
import os
from munch import Munch
from copy import deepcopy
from fedlab.utils.serialization import SerializationTool
from trainer import SubsetSerialTrainer
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.dataset import SubsetSampler
import copy
import warnings
import multiprocessing
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
from model import train_alone,test_alone
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Nowplaying', help='dataset name: diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=1024, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')


# 联邦相关参数
parser.add_argument("--client_num", default=10, type=int, help="客户机总数")
parser.add_argument("--sample_ratio", default=1, type=float, help="取多少客户的参数做聚合")
parser.add_argument("--com_round", default=200, type=int, help="运行几轮")

opt = parser.parse_args()
print(opt)
federated_args = Munch
federated_args.total_client = opt.client_num
federated_args.sample_ratio = opt.sample_ratio
federated_args.com_round = opt.com_round

def _get_subdataset(client_id, train_data_size, test_data_size, client_size, train_dataset, test_dataset):
    scale_factor = 0.02
    # 按照客户机数量均分数据集
    train_piece_size = int(scale_factor*train_data_size//client_size)
    test_piece_size = int(scale_factor*test_data_size//client_size)
    print("train_piece_size:",train_piece_size)
    print("test_piece_size",test_piece_size)
    train_start_index =  train_piece_size * (client_id - 1)
    train_end_index = train_piece_size * (client_id - 1) + train_piece_size
    
    test_start_index =  test_piece_size * (client_id - 1)
    test_end_index = test_piece_size * (client_id - 1) + test_piece_size
    
    
    train_data=[train_dataset[0][train_start_index:train_end_index],train_dataset[1][train_start_index:train_end_index]]
   
    test_data = [test_dataset[0][test_start_index:test_end_index],test_dataset[1][test_start_index:test_end_index]]

    return train_data, test_data

def load_dataset(dataset):
    train_data = pickle.load(open(dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open(dataset + '/test.txt', 'rb'))
    print("train data len:",len(train_data[0]),len(train_data[1]))
    print("test data len:",len(test_data[0]),len(test_data[1]))
    return train_data, test_data

def load_model(model,file_name):
    original_state_dict = model.state_dict()
    # print(original_state_dict.keys())
    new_dict = torch.load(file_name)
    # print(new_dict.keys())
    for key in new_dict:
        original_state_dict[key]=new_dict[key]
    model.load_state_dict(original_state_dict)
    return model

hit5_res = []
hit10_res = []
hit20_res = []


    
def main():
    # a = datetime.datetime.now()
    # pool = multiprocessing.Pool(processes=5)
    
    res_file = open("test_frac_002_50_async_05_epoch2_test03.txt",'a+')
    total_train_data,total_test_data = load_dataset(opt.dataset)
    

    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'Nowplaying':
        n_node = 60416
    else:
        n_node = 309

    # 包含所有数据的集合
    train_data_total = Data(total_train_data, shuffle=True, n_node=n_node)
    test_data_total = Data(total_test_data, shuffle=False, n_node=n_node)
    # 中心
    center_model = trans_to_cpu(DHCN(adjacency=train_data_total.adjacency,n_node=n_node,lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset))
    # center_model = load_model(center_model, "output/fixed_init_model.pt")
    # center_model_parameters = SerializationTool.serialize_model(center_model)
    # torch.save(center_model.state_dict(), "output/fixed_init_model.pt")

    # print("center_model_parameters:",center_model_parameters)
    
    
    # 给每个客户分数据集
    id_list = range(1,opt.client_num+1)
    client_train_data_lst = []
    client_test_data_lst = []
    for id in id_list:
        client_train_data, client_test_data = _get_subdataset(client_id=id, train_data_size=len(total_train_data[0]), test_data_size = len(total_test_data[0]),client_size=opt.client_num, 
                                                              train_dataset=total_train_data, test_dataset = total_test_data)
        client_train_data = Data(client_train_data, shuffle=False, n_node=n_node)
        # client_test_data = Data(client_test_data, shuffle=False, n_node=n_node)
        client_train_data_lst.append(client_train_data)
        client_test_data_lst.append(client_test_data)

    # aggregator = Aggregators.fedavg_aggregate
    aggregator = Aggregators.fedasync_aggregate
    top_K = [5, 10, 20]
    
    
    client_model_lst = []
    for idx in id_list:
        # client_train_data, client_test_data = _get_subdataset(client_id=idx,train_dataset=train_data, test_dataset = test_data)
        # client_train_data = Data(client_train_data, shuffle=False, n_node=n_node)
        client_model_lst.append(copy.deepcopy(center_model))
    frac_lst = [0.5]
    # frac_lst = [1]
    # pre_total_loss = 0
    for frac in frac_lst:
        res_file.write("frac:"+str(frac)+"\n")
        m = max(int(frac * opt.client_num), 1)
        print("m:",m)
        choice_lst = np.random.choice(range(1,m+1), int(0.5*m), replace=False)
        for round in range(opt.com_round):
            print("start round:",str(round))
            res_file.write("start round:"+str(round+1)+'\n')
            
            param_list = []
            for idx in choice_lst:
                print(
                    "Starting training procedure of client [{}]".format(idx))
                print('-------------------------------------------------------')
                client_model = client_model_lst[idx-1]
                # client_model_parameters = SerializationTool.serialize_model(client_model)
                # 获取当前客户机的数据
                client_train_data, client_test_data = client_train_data_lst[idx-1], client_test_data_lst[idx-1]
                # print("client_train_data:",client_train_data)
                # for epoch in range(opt.epoch):
                client_epoch  = 0
                for i in range(opt.epoch):
                    print('idx:',idx,' epoch:', i)
                    print('-------------------------------------------------------')
                    
                    client_model, total_loss= train_alone(client_model, train_data=client_train_data)
                    # if abs(total_loss - pre_total_loss) < 1e-3:
                    #     print("client converge!:",epoch)
                    #     break
                    # else:
                    # pre_total_loss = total_loss
                # print("client converge!:",idx,client_epoch)
                # 把当前客户机训练完成的参数加入参数列表
                # print("test_res:",test_alone(client_model, test_data=client_test_data))
                # print("after train:",SerializationTool.serialize_model(client_model))
                param_list.append(SerializationTool.serialize_model(client_model))

            # print("param list:",param_list)
            aggregated_parameters = aggregator(SerializationTool.serialize_model(center_model),SerializationTool.serialize_model(client_model),0.5)
            # aggregated_parameters = aggregator(param_list)
            # print("aggregated_parameters:",aggregated_parameters)
            #  把聚合后的参数赋值给聚合服务器
            SerializationTool.deserialize_model(center_model,aggregated_parameters)

            #  把聚合后的参数赋值给每个客户机
            for idx in choice_lst:
                    client_model = client_model_lst[idx-1]
                    SerializationTool.deserialize_model(client_model,aggregated_parameters)
                    client_model_lst[idx-1] = client_model
            # tmp_hit5 = 0
            # tmp_hit10 = 0
            # tmp_hit20 = 0
            # 测试当前全局模型的效果
            # for idx in choice_lst:
            avg_matrics={}
            for K in top_K:
                avg_matrics['hit%d' % K] = 0
                avg_matrics['mrr%d' % K] = 0

            rs = [0.3,0.3,0.3]
            for j in rs:
                metrics = test_alone(center_model, test_data =test_data_total, rs = j)
                for K in top_K:
                    metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
                    metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
                    avg_matrics['hit%d' % K] += metrics['hit%d' % K]
                    avg_matrics['mrr%d' % K] += metrics['mrr%d' % K]
            
                # best_results = {}
                # for K in top_K:
                #     best_results['round%d' % K] = [0, 0]
                #     best_results['metric%d' % K] = [0, 0]

                
                #     if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                #         best_results['metric%d' % K][0] = metrics['hit%d' % K]
                #         best_results['round%d' % K][0] = round
                #     if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                #         best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                #         best_results['round%d' % K][1] = round
                
                # print("center_model metric:",best_results)
            for K in top_K:
                avg_matrics['hit%d' % K] = avg_matrics['hit%d' % K]/3
                avg_matrics['mrr%d' % K] = avg_matrics['mrr%d' % K]/3
                print('center_model \tRecall@%d: %.4f\tMRR%d: %.4f\t' %
                    ( K, avg_matrics['hit%d' % K], K, avg_matrics['mrr%d' % K]))
            
            res_file.write(str(avg_matrics)+"\n")
            res_file.write('------------------------------\n')

           
            
            
    # res_file.write(str(hit5_res)+'\n')
    # res_file.write(str(hit10_res)+'\n')
    # res_file.write(str(hit20_res)+'\n')
    res_file.close()
    


if __name__ == '__main__':
    main()


