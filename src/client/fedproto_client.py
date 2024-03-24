from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn
import torch
import copy
import math
criterion = F.cross_entropy

mse_loss = nn.MSELoss()
from src.utils.torch_utils import *
from .base_client import Client



class FedProtoClient(Client):
    def __init__(self, options, id, model, optimizer, local_dataset):

        super(FedProtoClient, self).__init__(options, id, model, optimizer, local_dataset)
        self.local_protos = None


    def local_update(self, local_dataset, options, global_protos):
        # batch_size=options['batch_size']
        if options['batch_size'] == -1:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
        else:
            if len(local_dataset) < options['batch_size']:
                localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
            else:
                sampler = RandomSampler(local_dataset, replacement=False, num_samples=1 * options['batch_size'])
                localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], sampler=sampler)
                # localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        agg_protos_label = {}
        self.model.train()
        train_loss = train_acc = train_total = 0
        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                # print("y", y)
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                self.optimizer.zero_grad()
                feature, pred = self.model(X)
                loss_s = criterion(pred, y)
                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss_r = 0 * loss_s
                # ------------------------------------ #
                else:
                    params = []
                    params.append(feature.data.view(len(y), -1))
                    feature_flat = torch.cat(params)
                    # print(feature_flat)
                    i = 0
                    for y_ in y:
                        if y_.item() in global_protos.keys():
                            feature_flat[i, :] = global_protos[y_.item()][0].data
                        i += 1
                    loss_r = loss_mse(feature_flat, feature)
                loss = loss_s + 1 * loss_r
                loss.backward()
                self.optimizer.step()
                for i in range(len(y)):
                    if y[i].item() in agg_protos_label:
                        agg_protos_label[y[i].item()].append(feature[i, :])
                    else:
                        agg_protos_label[y[i].item()] = [feature[i, :]]
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
        local_model_paras = self.get_flat_model_params()
        # print(local_model_paras)
        return_dict = {"id": self.id,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return local_model_paras, return_dict, agg_protos_label

    def local_train(self, global_protos):
        begin_time = time.time()
        local_model_paras, dict, agg_protos_label = self.local_update(self.local_dataset, self.options, global_protos)
        end_time = time.time()
        stats = {'id': self.id, "time": round(end_time - begin_time, 2)}
        stats.update(dict)
        return (len(self.local_dataset), local_model_paras), stats, agg_protos_label
