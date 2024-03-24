from src.server.base_server import BaseFederated
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
from tqdm import tqdm
from torch.optim import SGD, Adam
import copy
from src.client.fedproto_client import FedProtoClient
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch


class FedProto(BaseFederated):
    def __init__(self, options, dataset, clients_label, ):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr'])  # , weight_decay=0.001
        super(FedProto, self).__init__(options, dataset, clients_label, model, self.optimizer, )
        self.global_protos = {}

    def train(self):
        print('=== Select {} clients per round ===\n'.format(int(self.per_round_c_fraction * self.clients_num)))

        for round_i in range(self.num_round):
            self.test_latest_model_on_testdata(round_i)

            selected_clients = self.select_clients()

            local_model_paras_set, stats, agg_protos_label_es = self.local_train(round_i, selected_clients, self.global_protos)
            # print(agg_protos_label_es)
            self.global_protos = self.update_global_protos(agg_protos_label_es)
            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)
            self.optimizer.soft_decay_learning_rate()

        self.test_latest_model_on_testdata(self.num_round)

        self.metrics.write()

    def update_global_protos(self, agg_protos_label_es):
        global_protos = dict()
        for k in agg_protos_label_es:
            temp = torch.zeros_like(agg_protos_label_es[k][0])
            for tensor in agg_protos_label_es[k]:
                temp += tensor
            temp /= len(agg_protos_label_es[k])
            global_protos[k] = temp
        return global_protos


    def select_clients(self):
        num_clients = min(int(self.per_round_c_fraction * self.clients_num), self.clients_num)
        index = np.random.choice(len(self.clients), num_clients, replace=False, )
        select_clients = []
        for i in index:
            select_clients.append(self.clients[i])
        return select_clients

    def setup_clients(self, dataset, clients_label):
        train_data = dataset.trainData
        train_label = dataset.trainLabel
        all_client = []
        for i in range(len(clients_label)):
            local_client = FedProtoClient(self.options, i, self.model, self.optimizer,
                                  TensorDataset(torch.tensor(train_data[self.clients_label[i]]),
                                                torch.tensor(train_label[self.clients_label[i]])))
            all_client.append(local_client)

        return all_client

    def local_train(self, round_i, select_clients, global_protos):
        local_model_paras_set = []
        stats = []
        agg_protos_label_es = {}
        for i, client in enumerate(select_clients, start=1):
            client.set_flat_model_params(self.latest_global_model)

            local_model_paras, stat, agg_protos_label = client.local_train(global_protos)
            local_model_paras_set.append(local_model_paras)
            stats.append(stat)
            # print(agg_protos_label)
            for k in agg_protos_label:
                if k in agg_protos_label_es:
                    agg_protos_label_es[k].extend(agg_protos_label[k])
                else:
                    agg_protos_label_es[k] = agg_protos_label[k]
            if True:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                    round_i, client.id, i, int(self.per_round_c_fraction * self.clients_num),
                    stat['loss'], stat['acc'] * 100, stat['time'], ))
        return local_model_paras_set, stats, agg_protos_label_es
