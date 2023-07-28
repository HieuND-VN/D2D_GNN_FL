import numpy as np
import random
import torch
from torch_geometric.data import Data



class Env():
    def __init__(self, args):
        self.num_train = args.num_train
        self.num_test = args.num_test

        # Device information
        self.num_ap = args.num_clients
        self.num_ue_case1 = []
        for i in range(self.num_ap):
            self.num_ue_case1.append(random.randrange(args.num_ue_min, args.num_ue_max + 1, 10))
        # print(f'NUMBER OF ENV: {self.num_ue_case1}')
        self.num_ue_case2 = args.num_ue_min  # 10
        self.num_ue_case3 = 20
        self.num_ue_case4 = 30
        self.num_ue_case5 = 40
        self.num_ue_case6 = 50
        self.num_ue_case7 = args.num_ue_max  # 60
        self.diameter = args.diameter


        # Channel information
        self.var_db = args.var_db
        self.var_noise = 1 / 10 ** (self.var_db / 10)
        self.interference = args.interference
        self.bandwidth = args.bandwidth
        self.ha = args.height_ap
        self.hm = args.height_mobile
        self.freq = args.frequency
        self.power_f = args.power_f
        self.ther_noise = args.ther_noise
        self.d0 = args.distance_0
        self.d1 = args.distance_1
        self.is_train = True
        self.batch_size = args.batch_size

        self.aL = (1.1 * np.log10(self.freq) - 0.7) * self.hm - (1.56 * np.log10(self.freq) - 0.8)
        self.L = 46.3 + 33.9 * np.log10(self.freq) - 13.82 * np.log10(self.ha) - self.aL

        self.Pd = 1 / self.ther_noise
        self.Pu = self.Pd

    def create_graph_data(self, id, is_train=False, case=1):
        # select num_sample
        if is_train:
            num_sample = self.num_train
        else:
            num_sample = self.num_test
        # select num_ue in each cell
        if case == 1:
            num_ue = self.num_ue_case1[id]
        elif case == 2:
            num_ue = self.num_ue_case2
        elif case == 3:
            num_ue = self.num_ue_case3
        elif case == 4:
            num_ue = self.num_ue_case4
        elif case == 5:
            num_ue = self.num_ue_case5
        elif case == 6:
            num_ue = self.num_ue_case5

        data_client = self.proc_data(num_sample, num_ue)
        return 3h data_client

    def proc_data(self, num_sample, num_ue):
        data_list = []
        X, Y, A, Y2 = self.generate_wGaussian(num_ue, num_sample)
        cg = self.get_cg(num_ue)
        for sample in range(num_sample):
            data = self.build_graph(X[sample], A[sample], cg, num_ue)
            data_list.append(data)
        return data_list

    def build_graph(self, H, A, adj, num_ue):
        x1 = np.expand_dims(np.diag(H), axis=1)
        x2 = np.expand_dims(A, axis=1)
        x3 = np.ones((num_ue, 1))
        edge_attr = []

        x = np.concatenate((x1, x2, x3), axis=1)
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

    def generate_wGaussian(self, num_ue, num_sample):
        Pmax = 1
        Pini = Pmax * np.ones((num_sample, num_ue, 1))
        alpha = np.random.rand(num_sample, num_ue)
        alpha2 = np.ones((num_sample, num_ue))
        CH = 1 / np.sqrt(2) * (
                    np.random.rand(num_sample, num_ue, num_ue) + 1j * np.random.rand(num_sample, num_ue, num_ue))
        H = abs(CH)
        Y = self.batch_WMMSE(Pini, alpha, H, Pmax)
        Y2 = self.batch_WMMSE(Pini, alpha2, H, Pmax)
        return H, Y, alpha, Y2

    def batch_WMMSE(self, p_int, alpha, H, Pmax):
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

    def get_cg(self, num_user):
        adj = []
        for i in range(0, num_user):
            for j in range(0, num_user):
                if (not (i == j)):
                    adj.append([i, j])
        return adj

    # def simple_greedy(self, X, AAA, label, num_user):
    #     n = X.shape[0]
    #     thd = int(np.sum(label) / n)
    #     Y = np.zeros((n, num_user))
    #     for ii in range(n):
    #         alpha = AAA[ii, :]
    #         H_diag = alpha * np.square(np.diag(X[ii, :, :]))
    #         xx = np.argsort(H_diag)[::-1]
    #         for jj in range(thd):
    #             Y[ii, xx[jj]] = 1
    #     return Y
    #
    # def np_sum_rate(self, H, p, alpha, var_noise):
    #     H = np.expand_dims(H, axis=-1)
    #     K = H.shape[1]
    #     N = H.shape[-1]
    #     p = p.reshape((-1, K, 1, N))
    #     rx_power = np.multiply(H, p)
    #     rx_power = np.sum(rx_power, axis=-1)
    #     rx_power = np.square(abs(rx_power))
    #     mask = np.eye(K)
    #     valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=1)
    #     interference = np.sum(np.multiply(rx_power, 1 - mask), axis=1) + var_noise
    #     rate = np.log(1 + np.divide(valid_rx_power, interference))
    #     w_rate = np.multiply(alpha, rate)
    #     sum_rate = np.mean(np.sum(w_rate, axis=1))
    #     return sum_rate
    #
    # def show_result_graph(self, num_ue, num_test, var_db, num_client):
    #     seed = testseed
    #     num_users = num_user * num_client
    #     num_tests = num_test * num_client
    #     var_noise = 1 / 10 ** (var_db / 10)
    #     Xtest, Ytest, Atest, Ytest2 = generate_wGaussian(num_users, num_tests, var_noise, seed)
    #     baseline_Y = self.simple_greedy(Xtest, Atest, Ytest, num_users)
    #     print(f'Greedy - Baseline Methods:  {self.np_sum_rate(Xtest, baseline_Y, Atest, var_noise)}')
    #     print(f'WMMSE - Weighted Case:      {self.np_sum_rate(Xtest.transpose(0, 2, 1), Ytest, Atest, var_noise)}')
    #     print(f'WMMSE - Unweighted Case:    {self.np_sum_rate(Xtest.transpose(0, 2, 1), Ytest2, Atest, var_noise)}')


