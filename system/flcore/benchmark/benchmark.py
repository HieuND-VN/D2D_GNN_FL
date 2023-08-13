import torch_geometric
import numpy as np
import torch
import torch.nn as nn
from flcore.trainmodel.models import IGCNet

class Benchmark10():
    def __init__(self, args, trainloader_10, testloader_10, testloader_50, testloader_100):
        # Model
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.global_rounds = args.global_rounds
        self.device = args.device
        self.model = IGCNet().to(self.device)
        self.learning_rate = args.local_learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=args.step_size,
            gamma=args.learning_rate_decay_gamma)
        self.learning_rate_decay = args.learning_rate_decay
        self.trainloader = trainloader_10
        self.testloader_10 = testloader_10
        self.testloader_50 = testloader_50
        self.testloader_100 = testloader_100
        self.var_noise = 1/10 ** (args.var_db/10)
        self.loss_train = []
        self.loss_test_10 = []
        self.loss_test_50 = []
        self.loss_test_100 = []

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

    def calculate(self, is_print = False):
        for step in range(self.global_rounds):
            total_loss_train = 0
            self.model.train()
            for data in self.trainloader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(data)
                # loss = self.sr_loss(data, out, K = 10)
                loss = self.sr_loss(data, out, K=2)
                loss.backward()
                total_loss_train += loss.item()*data.num_graphs
                self.optimizer.step()
            loss_tr = (total_loss_train / self.num_train)*-1

            self.model.eval()
            total_loss_te10 = 0
            total_loss_te50 = 0
            total_loss_te100 = 0
            for data_10 in self.testloader_10:
                data_10 = data_10.to(self.device)
                with torch.no_grad():
                    out_10 = self.model(data_10)
                    # loss_10 = self.sr_loss(data_10, out_10, K = 10)
                    loss_10 = self.sr_loss(data_10, out_10, K=2)
                    total_loss_te10 += loss_10.item()*data_10.num_graphs
            loss_te10 = (total_loss_te10/self.num_test)*-1

            for data_50 in self.testloader_50:
                data_50 = data_50.to(self.device)
                with torch.no_grad():
                    out_50 = self.model(data_50)
                    # loss_50 = self.sr_loss(data_50, out_50, K = 50)
                    loss_50 = self.sr_loss(data_50, out_50, K=6)
                    total_loss_te50 += loss_50.item()*data_50.num_graphs
            loss_te50 = (total_loss_te50/self.num_test)*-1

            for data_100 in self.testloader_100:
                data_100 = data_100.to(self.device)
                with torch.no_grad():
                    out_100 = self.model(data_100)
                    # loss_100 = self.sr_loss(data_100, out_100, K = 100)
                    loss_100 = self.sr_loss(data_100, out_100, K=10)
                    total_loss_te100 += loss_100.item()*data_100.num_graphs
            loss_te100 = (total_loss_te100/self.num_test)*-1

            if (step%25 == 0) and (is_print):
                print('Epoch: {:03d}: [Train_10: {:.4f}] --- [Test_10: {:.4f}]  --- [Test_50: {:.4f}] --- [Test_100: {:.4f}]'.format(step, loss_tr, loss_te10, loss_te50, loss_te100))

            self.loss_train.append(loss_tr)
            self.loss_test_10.append(loss_te10)
            self.loss_test_50.append(loss_te50)
            self.loss_test_100.append(loss_te100)

        return self.loss_train, self.loss_test_10, self.loss_test_50, self.loss_test_100


class Benchmark50():
    def __init__(self, args, trainloader_50, testloader_10, testloader_50, testloader_100):
        # Model
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.global_rounds = args.global_rounds
        self.device = args.device
        self.model = IGCNet().to(self.device)
        self.learning_rate = args.local_learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=args.step_size,
            gamma=args.learning_rate_decay_gamma)
        self.learning_rate_decay = args.learning_rate_decay
        self.trainloader = trainloader_50
        self.testloader_10 = testloader_10
        self.testloader_50 = testloader_50
        self.testloader_100 = testloader_100
        self.var_noise = 1 / 10 ** (args.var_db / 10)
        self.loss_train = []
        self.loss_test_10 = []
        self.loss_test_50 = []
        self.loss_test_100 = []

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

    def calculate(self, is_print = False):
        for step in range(self.global_rounds):
            total_loss_train = 0
            self.model.train()
            for data in self.trainloader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(data)
                # loss = self.sr_loss(data, out, K=50)
                loss = self.sr_loss(data, out, K=6)
                loss.backward()
                total_loss_train += loss.item() * data.num_graphs
                self.optimizer.step()
            loss_tr = (total_loss_train / self.num_train) * -1

            self.model.eval()
            total_loss_te10 = 0
            total_loss_te50 = 0
            total_loss_te100 = 0
            for data_10 in self.testloader_10:
                data_10 = data_10.to(self.device)
                with torch.no_grad():
                    out_10 = self.model(data_10)
                    # loss_10 = self.sr_loss(data_10, out_10, K=10)
                    loss_10 = self.sr_loss(data_10, out_10, K=2)
                    total_loss_te10 += loss_10.item() * data_10.num_graphs
            loss_te10 = (total_loss_te10 / self.num_test) * -1

            for data_50 in self.testloader_50:
                data_50 = data_50.to(self.device)
                with torch.no_grad():
                    out_50 = self.model(data_50)
                    # loss_50 = self.sr_loss(data_50, out_50, K=50)
                    loss_50 = self.sr_loss(data_50, out_50, K=6)
                    total_loss_te50 += loss_50.item() * data_50.num_graphs
            loss_te50 = (total_loss_te50 / self.num_test) * -1

            for data_100 in self.testloader_100:
                data_100 = data_100.to(self.device)
                with torch.no_grad():
                    out_100 = self.model(data_100)
                    # loss_100 = self.sr_loss(data_100, out_100, K=100)
                    loss_100 = self.sr_loss(data_100, out_100, K=10)
                    total_loss_te100 += loss_100.item() * data_100.num_graphs
            loss_te100 = (total_loss_te100 / self.num_test) * -1

            if (step%25 == 0) and (is_print):
                print('[BASE_50] Epoch: {:03d}: [Train: {:.4f}] --- [Test_10: {:.4f}]  --- [Test_50: {:.4f}] --- [Test_100: {:.4f}]'.format(step, loss_tr, loss_te10, loss_te50, loss_te100))

            self.loss_train.append(loss_tr)
            self.loss_test_10.append(loss_te10)
            self.loss_test_50.append(loss_te50)
            self.loss_test_100.append(loss_te100)

        return self.loss_train, self.loss_test_10, self.loss_test_50, self.loss_test_100
class Benchmark100():
    def __init__(self, args, trainloader_100, testloader_10, testloader_50, testloader_100):
        # Model
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.global_rounds = args.global_rounds
        self.device = args.device
        self.model = IGCNet().to(self.device)
        self.learning_rate = args.local_learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=args.step_size,
            gamma=args.learning_rate_decay_gamma)
        self.learning_rate_decay = args.learning_rate_decay
        self.trainloader = trainloader_100
        self.testloader_10 = testloader_10
        self.testloader_50 = testloader_50
        self.testloader_100 = testloader_100
        self.var_noise = 1 / 10 ** (args.var_db / 10)
        self.loss_train = []
        self.loss_test_10 = []
        self.loss_test_50 = []
        self.loss_test_100 = []

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

    def calculate(self, is_print = False):
        for step in range(self.global_rounds):
            total_loss_train = 0
            self.model.train()
            for data in self.trainloader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(data)
                # loss = self.sr_loss(data, out, K=100)
                loss = self.sr_loss(data, out, K=10)
                loss.backward()
                total_loss_train += loss.item() * data.num_graphs
                self.optimizer.step()
            loss_tr = (total_loss_train / self.num_train) * -1

            self.model.eval()
            total_loss_te10 = 0
            total_loss_te50 = 0
            total_loss_te100 = 0
            for data_10 in self.testloader_10:
                data_10 = data_10.to(self.device)
                with torch.no_grad():
                    out_10 = self.model(data_10)
                    # loss_10 = self.sr_loss(data_10, out_10, K=10)
                    loss_10 = self.sr_loss(data_10, out_10, K=2)
                    total_loss_te10 += loss_10.item() * data_10.num_graphs
            loss_te10 = (total_loss_te10 / self.num_test) * -1

            for data_50 in self.testloader_50:
                data_50 = data_50.to(self.device)
                with torch.no_grad():
                    out_50 = self.model(data_50)
                    # loss_50 = self.sr_loss(data_50, out_50, K=50)
                    loss_50 = self.sr_loss(data_50, out_50, K=6)
                    total_loss_te50 += loss_50.item() * data_50.num_graphs
            loss_te50 = (total_loss_te50 / self.num_test) * -1

            for data_100 in self.testloader_100:
                data_100 = data_100.to(self.device)
                with torch.no_grad():
                    out_100 = self.model(data_100)
                    # loss_100 = self.sr_loss(data_100, out_100, K=100)
                    loss_100 = self.sr_loss(data_100, out_100, K=10)
                    total_loss_te100 += loss_100.item() * data_100.num_graphs
            loss_te100 = (total_loss_te100 / self.num_test) * -1

            if (step%25 == 0) and (is_print):
                print('Epoch: {:03d}: [Train_10: {:.4f}] --- [Test_10: {:.4f}]  --- [Test_50: {:.4f}] --- [Test_100: {:.4f}]'.format(step, loss_tr, loss_te10, loss_te50, loss_te100))

            self.loss_train.append(loss_tr)
            self.loss_test_10.append(loss_te10)
            self.loss_test_50.append(loss_te50)
            self.loss_test_100.append(loss_te100)

        return self.loss_train, self.loss_test_10, self.loss_test_50, self.loss_test_100
