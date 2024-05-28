import pickle

class DatasetSplit():
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        train_data, test_data = self.dataset[self.idxs[item]]
        return 
    


def load_dataset(dataset):
    train_data = pickle.load(open(dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open(dataset + '/test.txt', 'rb'))
    print("train data len:",len(train_data[0]),len(train_data[1]))
    print("test data len:",len(test_data[0],len(test_data[1])))
    return train_data, test_data



