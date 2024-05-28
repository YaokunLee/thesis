import json
import matplotlib.pyplot as plt
import csv
file_handler = open("res_epoch1_seed.txt",'r')

file_str = file_handler.read()
lst = file_str.split("------------------------------\n")
lst_len = len(lst) - 1
hit5_lst = []
hit10_lst = []
hit20_lst = []
mrr5_lst = []
mrr10_lst = []
mrr20_lst = []

file_handler2 = open("/opt/distributed_v2/DHT+Flask/node1/DHCN/res_record_seed.txt",'r')
file_str2 = file_handler2.read()
lst2 = file_str2.split("------------------------------\n")
lst_len2 = len(lst2) - 1
hit5_lst2 = []
hit10_lst2 = []
hit20_lst2 = []
mrr5_lst2 = []
mrr10_lst2 = []
mrr20_lst2 = []

for i in range(lst_len):
    lst[i]= lst[i].split('\n')[1]
    hit5 = float(lst[i].split("'hit5': ")[1].split(',')[0])
    hit10 = float(lst[i].split("'hit10': ")[1].split(',')[0])
    hit20 = float(lst[i].split("'hit20': ")[1].split(',')[0])
    mrr5= float(lst[i].split("'mrr5': ")[1].split(',')[0])
    mrr10 = float(lst[i].split("'mrr10': ")[1].split(',')[0])
    mrr20 = float(lst[i].split("'mrr20': ")[1].split('}')[0])
    hit5_lst.append(hit5)
    hit10_lst.append(hit10)
    hit20_lst.append(hit20)
    mrr5_lst.append(mrr5)
    mrr10_lst.append(mrr10)
    mrr20_lst.append(mrr20)
    # s = lst[i].split("'hit5:'")
    # print(s)
    # hit5 = s["hit5"]
    # hit10 = s["hit10"]
    # hit20 = s["hit20"]
    # mrr5= s["mrr5"]
    # mrr10 = s["mrr10"]
    # mrr20 = s["mrr20"]
    # print("hit5:",hit5)
    # print("mrr5",mrr5)
for i in range(lst_len2):
    lst2[i]= lst2[i].split('\n')[1]
    hit5 = float(lst2[i].split("'hit5': ")[1].split(',')[0])
    hit10 = float(lst2[i].split("'hit10': ")[1].split(',')[0])
    hit20 = float(lst2[i].split("'hit20': ")[1].split(',')[0])
    mrr5= float(lst2[i].split("'mrr5': ")[1].split(',')[0])
    mrr10 = float(lst2[i].split("'mrr10': ")[1].split(',')[0])
    mrr20 = float(lst2[i].split("'mrr20': ")[1].split('}')[0])
    hit5_lst2.append(hit5)
    hit10_lst2.append(hit10)
    hit20_lst2.append(hit20)
    mrr5_lst2.append(mrr5)
    mrr10_lst2.append(mrr10)
    mrr20_lst2.append(mrr20)

data1 = {}
data2 ={}
data3 ={}
data4={}
for i in range(1,101):
    data1[i] = hit5_lst[i-1]
    data2[i] = mrr5_lst[i-1]
    data3[i] = hit10_lst[i-1]
    data4[i] = mrr10_lst[i-1]

data2_1 = {}
data2_2 ={}
data2_3 ={}
data2_4={}
for i in range(1,101):
    data2_1[i] = hit5_lst2[i-1]
    data2_2[i] = mrr5_lst2[i-1]
    data2_3[i] = hit10_lst2[i-1]
    data2_4[i] = mrr10_lst2[i-1]
names2_1 = list(data2_1.keys())
values2_1 = list(data2_1.values())

names2_2 = list(data2_2.keys())
values2_2 = list(data2_2.values())

names2_3 = list(data2_3.keys())
values2_3 = list(data2_3.values())

names2_4 = list(data2_4.keys())
values2_4 = list(data2_4.values())

names1 = list(data1.keys())
values1 = list(data1.values())

names2 = list(data2.keys())
values2 = list(data2.values())

names3 = list(data3.keys())
values3 = list(data3.values())

names4 = list(data4.keys())
values4 = list(data4.values())

fig, axs = plt.subplots(1, 4, figsize=(9, 5), sharey=True)
# axs[0].bar(names, values)
axs[0].scatter(names1, values1,label = "fed")
axs[1].plot(names2, values2,label = "fed")
axs[2].plot(names3, values3,label = "fed")
axs[3].plot(names4, values4,label = "fed")

axs[0].scatter(names2_1, values2_1,label = "dis")
axs[1].plot(names2_2, values2_2,label = "dis")
axs[2].plot(names2_3, values2_3,label = "dis")
axs[3].plot(names2_4, values2_4,label = "dis")



axs[0].set_xlabel("round")
axs[0].set_ylabel("HIT@5")

axs[1].set_xlabel("round")
axs[1].set_ylabel("MRR@5")

axs[2].set_xlabel("round")
axs[2].set_ylabel("HIT@10")

axs[3].set_xlabel("round")
axs[3].set_ylabel("MRR@10")



fig.suptitle('Traditional federated learning')
plt.legend()
plt.savefig("res.png")
plt.show()


# res_file = open("federated_metrics.csv",'w')
# writer = csv.writer(res_file)
# writer.writerow(("HIT@5","HIT@10","HIT@20","MRR@5","MRR@10","MRR@20"))
# for i in range(1,101):
#     writer.writerow([hit5_lst[i-1],hit10_lst[i-1],hit20_lst[i-1],mrr5_lst[i-1],mrr10_lst[i-1],mrr20_lst[i-1]])
# res_file.close()

res_file = open("distributed_metrics.csv",'w')
writer = csv.writer(res_file)
writer.writerow(("HIT@5","HIT@10","HIT@20","MRR@5","MRR@10","MRR@20"))
for i in range(1,101):
    writer.writerow([hit5_lst2[i-1],hit10_lst2[i-1],hit20_lst2[i-1],mrr5_lst2[i-1],mrr10_lst2[i-1],mrr20_lst2[i-1]])
res_file.close()