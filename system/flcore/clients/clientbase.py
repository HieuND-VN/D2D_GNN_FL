import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
# from utils.data_utils import read_client_data
# from utils.data_utils import create_graph_data




class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, env, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        self.sample_rate = self.batch_size / self.num_train
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=args.step_size,
            gamma=args.learning_rate_decay_gamma)
        self.learning_rate_decay = args.learning_rate_decay

        self.num_ue = env.num_ue_array[self.id]
        self.train_data_loader_client = env.create_graph_data_train(id = self.id)
        self.test_data_loader_client = env.create_graph_data_test(num_user = self.num_ue)
        self.test_data_loader_10 = env.create_graph_data_test(num_user = 10)
        self.test_data_loader_50 = env.create_graph_data_test(num_user=50)
        self.test_data_loader_100 = env.create_graph_data_test(num_user=100)
        self.var_noise = env.var_noise

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()


    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        self.model.eval()
        total_loss_10 = 0
        total_loss_50 = 0
        total_loss_100 = 0
        total_loss = 0

        # self_test
        for data in self.test_data_loader_client:
            data = data.to(self.device)
            with torch.no_grad():
                out = self.model(data)
                loss = self.sr_loss(data, out, self.num_ue)
                total_loss += loss.item() * data.num_graphs
        test_loss = total_loss / self.num_test

        #test_10 user
        for data_10 in self.test_data_loader_10:
            data_10 = data_10.to(self.device)
            with torch.no_grad():
                out_10 = self.model(data_10)
                loss_10 = self.sr_loss(data_10, out_10, 10)
                total_loss_10 += loss_10.item() * data_10.num_graphs
        test_loss_10 = total_loss_10 / self.num_test

        #test_50 user
        for data_50 in self.test_data_loader_50:
            data_50 = data_50.to(self.device)
            with torch.no_grad():
                out_50 = self.model(data_50)
                loss_50 = self.sr_loss(data_50, out_50, 50)
                total_loss_50 += loss_50.item() * data_50.num_graphs
        test_loss_50 = total_loss_50 / self.num_test

        #test_100 user
        for data_100 in self.test_data_loader_100:
            data_100 = data_100.to(self.device)
            with torch.no_grad():
                out_100 = self.model(data_100)
                loss_100 = self.sr_loss(data_100, out_100, 100)
                total_loss_100 += loss_100.item() * data_100.num_graphs
        test_loss_100 = total_loss_100 / self.num_test

        return test_loss, test_loss_10, test_loss_50, test_loss_100

    def train_metrics(self):
        self.model.eval()
        total_loss = 0
        train_client_loss = 0
        for data in self.train_data_loader_client:
            data = data.to(self.device)
            with torch.no_grad():
                out = self.model(data)
                loss = self.sr_loss(data, out, self.num_ue)
                total_loss += loss.item() * data.num_graphs
        train_client_loss = total_loss / self.num_train
        return train_client_loss

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def sr_loss(self, data, out, K):
        power = out[:, 2]
        power = torch.reshape(power, (-1, K, 1))
        abs_H = data.y
        abs_H_2 = torch.pow(abs_H, 2)
        rx_power = torch.mul(abs_H_2, power)
        mask = torch.eye(K)
        mask = mask.to(self.device)
        valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
        interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + self.var_noise
        rate = torch.log(1 + torch.div(valid_rx_power, interference))
        w_rate = torch.mul(data.pos, rate)
        sum_rate = torch.mean(torch.sum(w_rate, 1))
        loss = torch.neg(sum_rate)
        return loss