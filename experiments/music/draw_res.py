import matplotlib.pyplot as plt
# plt.style.use('_mpl-gallery')

hit20_res = [4.265850945494995,4.28698553948832,4.403781979977753,4.50055617352614,4.579532814238043,4.682981090100111]
# mrr20_res = [1.8341291594112803,1.8863635711983235, 1.974362596145684,1.9577235454174304,
#              1.9473436003066422,1.955891804746681,1.9605061484353279,1.9761865011923059,1.982677588536335, 2.003439036518177]

hit10_res = [3.727474972191324,3.7630700778642936,3.8209121245828697,3.888765294771969,3.947719688542825,4.050055617352614]
hit5_res = [3.2558398220244715,3.310344827586207,3.3637374860956615,3.4260289210233594,3.4627363737486094,3.5617352614015574]
# data = {'1': 10, 'oranges': 15, 'lemons': 5, 'limes': 20}
data1 = {}
data2 ={}
data3 = {}
frac_lst = [0.1,0.2,0.4,0.6,0.8,1]
for i in range(1,7):
    data1[i] = hit20_res[i-1]
    data2[i] = hit10_res[i-1]
    data3[i] = hit5_res[i-1]
names1 = list(data1.keys())
values1 = list(data1.values())

names2 = list(data2.keys())
values2 = list(data2.values())

names3 =  list(data3.keys())
values3 = list(data3.values())

fig, axs = plt.subplots()
axs.plot(frac_lst,values1,linewidth=2.0,color="orange",label="hit@20")
axs.plot(frac_lst,values2,linewidth=2.0,color="red",label="hit@10")
axs.plot(frac_lst,values3,linewidth=2.0,color="green",label="hit@5")

# axs[0].plot(frac_lst, values1)
# axs[1].plot(frac_lst, values2)
# axs[2].plot(frac_lst, values3)
axs.grid(True)
axs.set_ylim(3,5)
axs.set_xlim(0,1)
axs.set_xlabel("Percentage of Training Clients (10 in total)",fontsize="x-large")
axs.set_ylabel("Hit Ratio ",fontsize = "x-large")

# axs[1].set_xlabel("frac")
# axs[1].set_ylabel("HIT@10")
# plt.ylim([0,5])
# axs[2].set_xlabel("frac")
# axs[2].set_ylabel("HIT@5")

# plt.title('Hit ratio', fontsize = "xx-large")
plt.legend()
plt.show()
plt.savefig("res.png")

