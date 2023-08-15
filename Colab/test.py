import numpy as np
import random
import numpy as np
import random
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN

import matplotlib.pyplot as plt

class IGConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(IGConv, self).__init__(aggr='max', **kwargs)
        self.mlp1 = mlp1
        self.mlp2 = mlp2
    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp)
        return torch.cat([x[:,:-1],comb],dim=1)
    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg
    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1,self.mlp2)
class IGCNet10(torch.nn.Module):
    def __init__(self):
        super(IGCNet10, self).__init__()
        self.mlp1 = MLP([5, 16, 32])
        self.mlp2 = MLP([35, 16])
        self.mlp2 = Seq(*[self.mlp2,Seq(Lin(16, 1, bias = True), Sigmoid())])
        self.conv = IGConv(self.mlp1,self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)
        x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        return out

class IGCNet50(torch.nn.Module):
    def __init__(self):
        super(IGCNet50, self).__init__()
        self.mlp1 = MLP([5, 16, 32])
        self.mlp2 = MLP([35, 16])
        self.mlp2 = Seq(*[self.mlp2,Seq(Lin(16, 1, bias = True), Sigmoid())])
        self.conv = IGConv(self.mlp1,self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)
        x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        return out

class IGCNet100(torch.nn.Module):
    def __init__(self):
        super(IGCNet100, self).__init__()
        self.mlp1 = MLP([5, 16, 32])
        self.mlp2 = MLP([35, 16])
        self.mlp2 = Seq(*[self.mlp2,Seq(Lin(16, 1, bias = True), Sigmoid())])
        self.conv = IGConv(self.mlp1,self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)
        x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        return out

def MLP(channels, batch_norm=True):
    return Seq(*[Seq(Lin(channels[i - 1], channels[i], bias = True), ReLU())   for i in range(1, len(channels))])

class Env():
    def __init__(self):
        self.num_train = 3000
        self.num_test = 500
        # self.num_ue = self.num_ap = random.randrange(10, 61, 10)
        # self.num_ue = self.num_ap = 60
        self.bandwidth = 20
        self.diameter = 1
        self.var_db = 1
        self.var_noise = 1/10 ** (self.var_db / 10)
        self.interference = 0.5
        self.ha = 15
        self.hm = 1.65
        self.freq = 1900
        self.power_f = 0.2
        self.ther_noise = 20000000 * 10 ** (-17.4) * 10 ** -3
        self.d0 = 0.01
        self.d1 = 0.05
        self.is_train = True

        self.Pu = self.Pd = 1/self.ther_noise

        self.aL = (1.1 * np.log10(self.freq) - 0.7) * self.hm - (1.56 * np.log10(self.freq) - 0.8)
        self.L = 46.3 + 33.9 * np.log10(self.freq) - 13.82 * np.log10(self.ha) - self.aL

        self.greedy = 0
        self.weighted_case = 0
        self.unweighted_case = 0

    def create_graph_data(self, num_user, is_train = False):
        if is_train:
            num_sample = self.num_train
        else:
            num_sample = self.num_test

        num_ue = num_ap = num_user
        data_client = self.proc_data(num_sample, num_ue, num_ap)
        return data_client
        # if is_train:
        #   data_loader = DataLoader(data_client, batch_size=10, shuffle=True,num_workers=1)
        # else:
        #   data_loader = DataLoader(data_client, batch_size=10, shuffle=False, num_workers=1)
        # return data_loader

    def proc_data(self, num_sample, num_ue, num_ap):
        data_list = []
        X, Y, A, Y2 = self.generate_wGaussian(num_ue, num_sample, num_ap)
        cg = self.get_cg(num_ue)
        for sample in range(num_sample):
            data = self.build_graph(X[sample], A[sample], cg, num_ue)
            data_list.append(data)
        return data_list

    def generate_wGaussian(self, num_ue, num_sample, num_ap):
        Pmax = 1
        Pini = Pmax * np.ones((num_sample, num_ue, 1))
        alpha = np.random.rand(num_sample, num_ue)
        alpha2 = np.ones((num_sample, num_ue))

        CH = self.create_pathloss(num_ue, num_sample, num_ap)
        H = abs(CH)
        Y = self.batch_WMMSE(Pini, alpha, H, Pmax)
        Y2 = self.batch_WMMSE(Pini, alpha2, H, Pmax)
        return H, Y, alpha, Y2

    def get_cg(self, num_ue):
        adj = []
        for i in range(0, num_ue):
            for j in range(0, num_ue):
                if i!=j:
                    adj.append([i,j])
        return adj


    def create_pathloss(self, num_ue, num_sample, num_ap):
        pathloss = np.zeros((num_sample, num_ap, num_ue))
        for ite in range(num_sample):
            AP = np.random.uniform(0,0, size = (num_ap,2))
            UE = np.random.uniform(-1,1, size = (num_ue,2))
            pathloss_sample = np.zeros((num_ap, num_ue))
            for m in range(num_ap):
                for k in range(num_ue):
                    dist = np.linalg.norm(AP[m] - UE[k,:])
                    if dist < self.d0:
                        PL = -self.L - 35 * np.log10(self.d1) + 20 * np.log10(self.d1) - 20 * np.log10(self.d0)
                    elif dist >= self.d0 and dist <= self.d1:
                        PL = -self.L - 35 * np.log10(self.d1) + 20 * np.log10(self.d1) - 20 * np.log10(dist)
                    else:
                        PL = -self.L - 35 * np.log10(dist) + np.random.normal(0,1) * 7

                    pathloss_sample[m,k] = 10 ** (PL/10) * self.Pd
            pathloss[ite,:,:] = pathloss_sample
        return pathloss

    def batch_WMMSE(self, p_int, alpha , H, Pmax):
        N = p_int.shape[0]
        K = p_int.shape[1]
        b = np.sqrt(p_int)
        mask = np.eye(K)
        rx_power = np.multiply(H, b)
        rx_power_s = np.square(rx_power)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
        interference = np.sum(rx_power_s, 2) + self.var_noise
        f = np.divide(valid_rx_power, interference)
        w = 1 / (1 - np.multiply(f, valid_rx_power))

        for ii in range(100):
            fp = np.expand_dims(f, 1)
            rx_power = np.multiply(H.transpose(0, 2, 1), fp)
            valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
            bup = np.multiply(alpha, np.multiply(w, valid_rx_power))
            rx_power_s = np.square(rx_power)
            wp = np.expand_dims(w, 1)
            alphap = np.expand_dims(alpha, 1)
            bdown = np.sum(np.multiply(alphap, np.multiply(rx_power_s, wp)), 2)
            btmp = bup / bdown
            b = np.minimum(btmp, np.ones((N, K)) * np.sqrt(Pmax)) + np.maximum(btmp, np.zeros((N, K))) - btmp

            bp = np.expand_dims(b, 1)
            rx_power = np.multiply(H, bp)
            rx_power_s = np.square(rx_power)
            valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
            interference = np.sum(rx_power_s, 2) + self.var_noise
            f = np.divide(valid_rx_power, interference)
            w = 1 / (1 - np.multiply(f, valid_rx_power))
        p_opt = np.square(b)
        return p_opt

    def build_graph(self, H, A, adj, num_ue):
        x1 = np.expand_dims(np.diag(H), axis = 1)
        x2 = np.expand_dims(A, axis = 1)
        x3 = np.ones((num_ue,1))
        edge_attr = []

        x = np.concatenate((x1,x2,x3), axis = 1)
        for e in adj:
            edge_attr.append([H[e[0], e[1]], H[e[1], e[0]]])

        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(adj, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = torch.tensor(np.expand_dims(H, axis=0), dtype=torch.float)
        pos = torch.tensor(np.expand_dims(A, axis=0), dtype=torch.float)

        data = Data(x=x,
                    edge_index=edge_index.t().contiguous(),
                    edge_attr=edge_attr,
                    y=y,
                    pos=pos)

        return data

    def simple_greedy(self, X, AAA, label, num_user):
        n = X.shape[0]
        thd = int(np.sum(label) / n)
        Y = np.zeros((n, num_user))
        for ii in range(n):
            alpha = AAA[ii, :]
            H_diag = alpha * np.square(np.diag(X[ii, :, :]))
            xx = np.argsort(H_diag)[::-1]
            for jj in range(thd):
                Y[ii, xx[jj]] = 1
        return Y

    def np_sum_rate(self, H, p, alpha):
        H = np.expand_dims(H, axis=-1)
        K = H.shape[1]
        N = H.shape[-1]
        p = p.reshape((-1, K, 1, N))
        rx_power = np.multiply(H, p)
        rx_power = np.sum(rx_power, axis=-1)
        rx_power = np.square(abs(rx_power))
        mask = np.eye(K)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=1)
        interference = np.sum(np.multiply(rx_power, 1 - mask), axis=1) + self.var_noise
        rate = np.log(1 + np.divide(valid_rx_power, interference))
        w_rate = np.multiply(alpha, rate)
        sum_rate = np.mean(np.sum(w_rate, axis=1))
        return sum_rate


    def show_result_graph(self, num_ue):
        num_users = num_ap = num_ue
        num_tests = self.num_test
        Xtest, Ytest, Atest, Ytest2 = self.generate_wGaussian(num_users, num_tests, num_ap)
        baseline_Y = self.simple_greedy(Xtest, Atest, Ytest, num_users)
        self.greedy = self.np_sum_rate(Xtest, baseline_Y, Atest)
        self.weighted_case = self.np_sum_rate(Xtest.transpose(0, 2, 1), Ytest, Atest)
        self.unweighted_case = self.np_sum_rate(Xtest.transpose(0, 2, 1), Ytest2, Atest)
        print(f'Greedy[{num_ue}] - Baseline Methods:  {self.greedy}')
        print(f'WMMSE[{num_ue}] - Weighted Case:      {self.weighted_case}')
        print(f'WMMSE[{num_ue}] - Unweighted Case:    {self.unweighted_case}')
        return self.greedy, self.weighted_case, self.unweighted_case

def sr_loss(data, out, K):
    power = out[:,2]
    power = torch.reshape(power, (-1, K, 1))
    abs_H = data.y
    abs_H_2 = torch.pow(abs_H, 2)
    rx_power = torch.mul(abs_H_2, power)
    mask = torch.eye(K)
    mask = mask.to(device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + env.var_noise
    rate = torch.log(1 + torch.div(valid_rx_power, interference))
    w_rate = torch.mul(data.pos, rate)
    sum_rate = torch.mean(torch.sum(w_rate, 1))
    loss = torch.neg(sum_rate)
    return loss

if __name__ == "__main__":
    env = Env()

    train_data_loader_10 = env.create_graph_data(num_user=10, is_train=True)
    train_data_loader_50 = env.create_graph_data(num_user=50, is_train=True)
    train_data_loader_100 = env.create_graph_data(num_user=100, is_train=True)

    test_data_loader_10 = env.create_graph_data(num_user=10, is_train=False)
    test_data_loader_50 = env.create_graph_data(num_user=50, is_train=False)
    test_data_loader_100 = env.create_graph_data(num_user=100, is_train=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    greedy_10, WMMSE_weighted_10, WMMSE_unweighted_10 = env.show_result_graph(num_ue=10)
    print('=' * 50)
    greedy_50, WMMSE_weighted_50, WMMSE_unweighted_50 = env.show_result_graph(num_ue=50)
    print('=' * 50)
    greedy_100, WMMSE_weighted_100, WMMSE_unweighted_100 = env.show_result_graph(num_ue=100)

    model_10 = IGCNet10().to(device)
    optimizer_10 = torch.optim.Adam(model_10.parameters(), lr=0.001)
    scheduler_10 = torch.optim.lr_scheduler.StepLR(optimizer_10, step_size=20, gamma=0.9)
    # TRAIN 10, TEST 50, TEST 100

    loss_train_10 = []
    loss_train_10_test_10 = []
    loss_train_10_test_50 = []
    loss_train_10_test_100 = []
    # TRAIN_10
    for step in range(301):
        total_loss_train_10 = 0
        model_10.train()
        for data_10 in train_data_loader_10:
            if type(data_10) == type([]):
                data_10[0] = data_10[0].to(device)
            else:
                data_10 = data_10.to(device)
            optimizer_10.zero_grad()
            out_10 = model_10(data_10)
            loss_10 = sr_loss(data_10, out_10, K=10)
            loss_10.backward()
            total_loss_train_10 += loss_10.item() * data_10.num_graphs
            optimizer_10.step()
        loss_tr10 = (total_loss_train_10 / env.num_train) * -1

        model_10.eval()
        total_loss_tr10_te10 = 0
        total_loss_tr10_te50 = 0
        total_loss_tr10_te100 = 0
        # TEST_10
        for data_10 in test_data_loader_10:
            data_10 = data_10.to(device)
            with torch.no_grad():
                out_10 = model_10(data_10)
                loss_10 = sr_loss(data_10, out_10, K=10)
                total_loss_tr10_te10 += loss_10.item() * data_10.num_graphs
        loss_tr10_te10 = (total_loss_tr10_te10 / env.num_test) * -1
        # TEST_50
        for data_50 in test_data_loader_50:
            data_50 = data_50.to(device)
            with torch.no_grad():
                out_50 = model_10(data_50)
                loss_50 = sr_loss(data_50, out_50, K=50)
                total_loss_tr10_te50 += loss_50.item() * data_50.num_graphs
        loss_tr10_te50 = (total_loss_tr10_te50 / env.num_test) * -1
        # TEST_100
        for data_100 in test_data_loader_100:
            data_100 = data_100.to(device)
            with torch.no_grad():
                out_100 = model_10(data_100)
                loss_100 = sr_loss(data_100, out_100, K=100)
                total_loss_tr10_te100 += loss_100.item() * data_100.num_graphs
        loss_tr10_te100 = (total_loss_tr10_te100 / env.num_test) * -1
        loss_train_10.append(loss_tr10)
        loss_train_10_test_10.append(loss_tr10_te10)
        loss_train_10_test_50.append(loss_tr10_te50)
        loss_train_10_test_100.append(loss_tr10_te100)

        if step % 25 == 0:
            print(
                'Epoch: {:03d}: [Train_10: {:.4f}] --- [Test_10: {:.4f}]  --- [Test_50: {:.4f}] --- [Test_100: {:.4f}]'.format(
                    step, loss_tr10, loss_tr10_te10, loss_tr10_te50, loss_tr10_te100))

    loss_train_10 = np.array(loss_train_10)
    loss_train_10_test_10 = np.array(loss_train_10_test_10)
    loss_train_10_test_50 = np.array(loss_train_10_test_50)
    loss_train_10_test_100 = np.array(loss_train_10_test_100)

    x = np.arange(0, 302)
    # ==================================================================
    greedy_plot_10 = np.full_like(x, 1) * greedy_10
    WMMSE_weighted_plot_10 = np.full_like(x, 1) * WMMSE_weighted_10
    WMMSE_unweighted_plot_10 = np.full_like(x, 1) * WMMSE_unweighted_10
    # ==================================================================
    greedy_plot_50 = np.full_like(x, 1) * greedy_50
    WMMSE_weighted_plot_50 = np.full_like(x, 1) * WMMSE_weighted_50
    WMMSE_unweighted_plot_50 = np.full_like(x, 1) * WMMSE_unweighted_50
    # ==================================================================
    greedy_plot_100 = np.full_like(x, 1) * greedy_100
    WMMSE_weighted_plot_100 = np.full_like(x, 1) * WMMSE_weighted_100
    WMMSE_unweighted_plot_100 = np.full_like(x, 1) * WMMSE_unweighted_100
    # ==================================================================
    plt.plot(loss_train_10, label="TR = 10")
    plt.plot(loss_train_10_test_10, label="TE =  10")
    plt.plot(loss_train_10_test_50, label="TE =  50")
    plt.plot(loss_train_10_test_100, label="TE =  100")
    # ==================================================================
    # plt.plot(x,greedy_plot_10, label = 'Baseline_10')
    # plt.plot(x,WMMSE_unweighted_plot_10, label = 'Unweighted_10')
    plt.plot(x, WMMSE_weighted_plot_10, label='Weighted_10')
    # ==================================================================
    # plt.plot(x,greedy_plot_50, label = 'Baseline_50')
    # plt.plot(x,WMMSE_unweighted_plot_50, label = 'Unweighted_50')
    plt.plot(x, WMMSE_weighted_plot_50, label='Weighted_50')
    # ==================================================================
    # plt.plot(x,greedy_plot_100, label = 'Baseline_100')
    # plt.plot(x,WMMSE_unweighted_plot_100, label = 'Unweighted_100')
    plt.plot(x, WMMSE_weighted_plot_100, label='Weighted_100')
    # ==================================================================
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss_base10')
    plt.grid(linestyle='-.')
    plt.legend()
    plt.show()