import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as DLG
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.data_utils import create_graph_data

A = [['a'], ['b'], ['c'], ['a'], ['b'], ['c'], ['a'], ['b'], ['c']]


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
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
        self.sample_rate = self.batch_size / self.train_samples

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_Graph = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_Graph = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=args.step_size,
            gamma=args.learning_rate_decay_gamma)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma)
        self.learning_rate_decay = args.learning_rate_decay

        self.is_graph = args.is_graph
        self.num_train = args.train_samples
        self.num_test = args.test_samples
        self.num_client = args.num_clients
        self.num_user = args.num_user
        self.var_db = args.var_db
        self.var_noise = 1 / 10 ** (self.var_db / 10)
        self.train_loader = A[self.id]

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        if self.is_graph:
            train_data = create_graph_data(self.num_user,
                                           self.num_train,
                                           self.num_test,
                                           self.num_client,
                                           self.var_noise,
                                           self.id,
                                           is_train=True)
            return DLG(train_data, batch_size, shuffle=True, num_workers=1)
        else:
            train_data = read_client_data(self.dataset, self.id, is_train=True)
            return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        if self.is_graph:
            test_data = create_graph_data(self.num_user,
                                          self.num_train,
                                          self.num_test,
                                          self.num_client,
                                          self.var_noise,
                                          self.id,
                                          is_train=False)
            return DLG(test_data, batch_size, shuffle=False, num_workers=1)
        else:
            test_data = read_client_data(self.dataset, self.id, is_train=False)
            return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
            # print(f'New_param: {type(new_param.data)}-------{new_param.data.size()}')
            # print(f'Old_param: {type(old_param.data)}-------{old_param.data.size()}')

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        loss_test = 0

        with torch.no_grad():
            if self.is_graph:
                loss_test = self.test_metrics_graph(testloaderfull)
                return loss_test
            else:
                for x, y in testloaderfull:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.model(x)

                    test_acc += (torch.sum(torch.argmax(output,
                                                        dim=1) == y)).item()  # argmax: tim idx cua thang lon nhat. sau compare with label y
                    test_num += y.shape[0]

                    y_prob.append(output.detach().cpu().numpy())
                    nc = self.num_classes
                    if self.num_classes == 2:
                        nc += 1
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        lb = lb[:, :2]
                    y_true.append(lb)

                # self.model.cpu()
                # self.save_model(self.model, 'model')

                y_prob = np.concatenate(y_prob, axis=0)
                y_true = np.concatenate(y_true, axis=0)

                auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
                # print(f'Not phai Graph Here!')

                return test_acc, test_num, auc

    def train_metrics(self):
        # At train_metrics, we don't need to backward gradient descent!
        trainloaderfull = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            if self.is_graph:
                loss_train = self.train_metrics_graph(trainloaderfull)
                return loss_train
            else:
                for x, y in trainloaderfull:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.model(x)
                    loss = self.loss(output, y)
                    train_num += y.shape[0]
                    losses += loss.item() * y.shape[0]

            return losses, train_num

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

    def test_metrics_graph(self, test_loader):
        self.model.eval()
        total_loss = 0
        for data in test_loader:
            data = data.to(self.device)
            with torch.no_grad():
                out = self.model(data)
                loss = self.sr_loss(data, out, self.num_user)
                total_loss += loss.item() * data.num_graphs
        return total_loss / self.num_test

    def train_metrics_graph(self, train_loader):
        self.model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(self.device)
            out = self.model(data)
            loss = self.sr_loss(data, out, self.num_user)
            total_loss += loss.item() * data.num_graphs
        return total_loss / self.num_train

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