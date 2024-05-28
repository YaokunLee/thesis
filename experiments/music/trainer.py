# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)
import numpy as np
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from fedlab.core.client import ClientTrainer, SERIAL_TRAINER
from fedlab.utils import Logger
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.dataset import SubsetSampler
from torch.optim import Adam
from torch.utils.data import dataloader
import tqdm
import datetime
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class SerialTrainer(ClientTrainer):
    """Base class. Train multiple clients in sequence with a single process.

    Args:
        model (torch.nn.Module): Model used in this federation.
        client_num (int): Number of clients in current trainer.
        aggregator (Aggregators, callable, optional): Function to perform aggregation on a list of serialized model parameters.
        cuda (bool): Use GPUs or not. Default: ``True``.
        logger (Logger, optional): object of :class:`Logger`.
    """

    def __init__(self,
                 model,
                 client_num,
                 aggregator=None,
                 cuda=True,
                 logger=Logger()):
        super().__init__(model, cuda)
        self.client_num = client_num
        self.type = SERIAL_TRAINER  # represent serial trainer
        self.aggregator = aggregator
        self._LOGGER = logger

    def _train_alone(self, model_parameters, train_loader):
        """Train local model with :attr:`model_parameters` on :attr:`train_loader`.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters of one model.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        raise NotImplementedError()

    def _get_dataloader(self, client_id):
        """Get :class:`DataLoader` for ``client_id``."""
        raise NotImplementedError()

    def train(self, model_parameters, id_list, aggregate=False):
        """Train local model with different dataset according to client id in ``id_list``.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
            id_list (list[int]): Client id in this training serial.
            aggregate (bool): Whether to perform partial aggregation on this group of clients' local models at the end of each local training round.

        Note:
            Normally, aggregation is performed by server, while we provide :attr:`aggregate` option here to perform
            partial aggregation on current client group. This partial aggregation can reduce the aggregation workload
            of server.

        Returns:
            Serialized model parameters / list of model parameters.
        """
        param_list = []
        self._LOGGER.info(
            "Local training with client id list: {}".format(id_list))
        for idx in id_list:
            self._LOGGER.info(
                "Starting training procedure of client [{}]".format(idx))

            data_loader = self._get_dataloader(client_id=idx)
            # print("data_loader:",data_loader.sampler)
            # print("start_model_parameters:",model_parameters)
            self._train_alone(model_parameters=model_parameters,
                              train_loader=data_loader)
            # print("training.. (self.model_parameters):",self.model_parameters)
            # print("_model_parameter:",self.model_parameters)
            param_list.append(self.model_parameters)
        # print("param list:",param_list)
        if aggregate is True and self.aggregator is not None:
            # aggregate model parameters of this client group
            aggregated_parameters = self.aggregator(param_list)
            return aggregated_parameters
        else:
            return param_list


class SubsetSerialTrainer(SerialTrainer):
    """Train multiple clients in a single process.

    Customize :meth:`_get_dataloader` or :meth:`_train_alone` for specific algorithm design in clients.

    Args:
        model (torch.nn.Module): Model used in this federation.
        dataset (torch.utils.data.Dataset): Local dataset for this group of clients.
        data_slices (list[list]): subset of indices of dataset.
        aggregator (Aggregators, callable, optional): Function to perform aggregation on a list of model parameters.
        logger (Logger, optional): object of :class:`Logger`.
        cuda (bool): Use GPUs or not. Default: ``True``.
        args (dict, optional): Uncertain variables.

    .. note::
        ``len(data_slices) == client_num``, that is, each sub-index of :attr:`dataset` corresponds to a client's local dataset one-by-one.
    """

    def __init__(self,
                 model,
                 dataset,
                 data_slices,
                 aggregator=None,
                 logger=Logger(),
                 cuda=True,
                 args=None) -> None:

        super(SubsetSerialTrainer, self).__init__(model=model,
                                                  client_num=len(data_slices),
                                                  cuda=cuda,
                                                  aggregator=aggregator,
                                                  logger=logger)

        self.dataset = dataset
        self.data_slices = data_slices  # [0, client_num)
        self.args = args
        self.device = "cpu"
        # self.args.log_file = "output/log.txt"
        # self.optim = Adam(self.model.parameters(), lr=0.001, betas=(0.9,0.99), weight_decay=0.0)
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.betas=betas
        # self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

    def _get_dataloader(self, client_id):
        """Return a training dataloader used in :meth:`train` for client with :attr:`id`

        Args:
            client_id (int): :attr:`client_id` of client to generate dataloader

        Note:
            :attr:`client_id` here is not equal to ``client_id`` in global FL setting. It is the index of client in current :class:`SerialTrainer`.

        Returns:
            :class:`DataLoader` for specific client's sub-dataset
        """
        batch_size = self.args.batch_size
        # print(self.data_slices[client_id])
        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[client_id],
                                  shuffle=False),
            batch_size=batch_size)
        
        # train_dataset = FMLPRecDataset(self.args,self.args.seq_dic['user_seq'], data_type='session')
        # train_sampler=SubsetSampler(indices=self.data_slices[client_id],
        #                           shuffle=True)
        # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        return train_loader

    def _train_alone(self, model_parameters, train_data,batch_size):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        print('start training: ', datetime.datetime.now())
        torch.autograd.set_detect_anomaly(True)
        total_loss = 0.0
        slices = train_data.generate_batch(batch_size)
        for i in slices:
            model.zero_grad()
            targets, scores, con_loss = forward(model, i, train_data)
            loss = model.loss_function(scores + 1e-8, targets)
            loss = loss + con_loss
            loss.backward()
    #        print(loss.item())
            model.optimizer.step()
            total_loss += loss
        print('\tLoss:\t%.3f' % total_loss)
        top_K = [5, 10, 20]
        metrics = {}
        for K in top_K:
            metrics['hit%d' % K] = []
            metrics['mrr%d' % K] = []
        print('start predicting: ', datetime.datetime.now())
        # epochs, lr = self.args.epochs, self.args.lr
        # # print("epoch:",epochs,"lr:",lr)
        # # print("model_parameters_train_alone:",model_parameters)
        # SerializationTool.deserialize_model(self._model, model_parameters)
        # optim = Adam(self._model.parameters(), lr=lr, betas=self.betas, weight_decay=self.args.weight_decay)
        # # criterion = self.cross_entropy
        # str_code="train"
        # rec_data_iter = tqdm.tqdm(enumerate(train_loader),
        #                           desc="Recommendation EP_%s:%d" % (str_code, epochs),
        #                           total=len(train_loader),
        #                           bar_format="{l_bar}{r_bar}")
        # self._model.train()
        # rec_loss = 0.0
        # for e in range(epochs):
        #     # print("epochs:",e)
        #     for i, batch in rec_data_iter:
        #         # print("i:",i)
        #         # 0. batch_data will be sent into the device(GPU or CPU)
        #         batch = tuple(t.to(self.device) for t in batch)
        #         # print("batch:",batch)
        #         _, input_ids, answer, neg_answer = batch
        #         # Binary cross_entropy
        #         sequence_output = self.model(input_ids)
        #         # print("sequence_output:",sequence_output.shape)

        #         loss = self.cross_entropy(sequence_output, answer, neg_answer)

        #         optim.zero_grad()
        #         loss.backward()
        #         optim.step()
        #         rec_loss += loss.item()

        # post_fix = {
        #     "epochs": epochs,
        #     "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
        # }
        # # return  SerializationTool.serialize_model(self._model)
        return self.model_parameters

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        # [batch hidden_size]
        #pos = pos_emb.view(-1, pos_emb.size(2))
        #neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out[:, -1, :] # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos_emb * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg_emb * seq_emb, -1)
        #istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.mean(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        )# / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def get_metric(pred_list, topk=10):
        NDCG = 0.0
        HIT = 0.0
        MRR = 0.0
        # [batch] the answer's rank
        for rank in pred_list:
            MRR += 1.0 / (rank + 1.0)
            if rank < topk:
                NDCG += 1.0 / np.log2(rank + 2.0)
                HIT += 1.0
        return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)
    
    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)
    
    def c_evaluate(self, epoch, dataloader, full_sort=False, train=True):
        str_code = "train"
        if train == False:
            str_code = "test" 
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, answer, neg_answer = batch
                # Binary cross_entropy
                sequence_output = self.model(input_ids)
                # print("sequence_output:",sequence_output.shape)

                loss = self.cross_entropy(sequence_output, answer, neg_answer)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()

        else:
            self.model.eval()
            # print("rec_data_iter:",rec_data_iter)
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, answers, _, sample_negs = batch
                # print("test_batch:",batch)
                # print("input_ids_test:",input_ids.shape)
                # print("answer:",answers)
                # print("sample_neg:",sample_negs.shape)
                recommend_output = self.model(input_ids)
                # print("recommend_output:",recommend_output)
                test_neg_items = torch.cat((answers.unsqueeze(-1), sample_negs), -1)
                # print("test_neg_items:",test_neg_items.shape)
                recommend_output = recommend_output[:, -1, :]
                # print("recommend_output:",recommend_output.shape)
                test_logits = self.predict_sample(recommend_output, test_neg_items)
                test_logits = test_logits.cpu().detach().numpy().copy()
                if i == 0:
                    pred_list = test_logits
                else:
                    pred_list = np.append(pred_list, test_logits, axis=0)

            return self.get_sample_scores(epoch, pred_list)

