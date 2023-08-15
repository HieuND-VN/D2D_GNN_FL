import numpy as np
import random
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader



class Env():
    def __init__(self, args):
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.batch_size = args.batch_size

        # Device information
        self.num_clients = args.num_clients
        self.num_ue_array = np.arange(args.num_ue_min, args.num_ue_max+1, 10)
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

        self.aL = (1.1 * np.log10(self.freq) - 0.7) * self.hm - (1.56 * np.log10(self.freq) - 0.8)
        self.L = 46.3 + 33.9 * np.log10(self.freq) - 13.82 * np.log10(self.ha) - self.aL

        self.Pd = 1 / self.ther_noise
        self.Pu = self.Pd

        self.greedy = []
        self.weighted_case = []
        self.unweighted_case = []
    def env_print(self):
        print("===========================System Model======================================")
        print(f"Number of mobile users in each client (each cell): {self.num_ue_array}")
        print(f"Bandwidth of the channel: {self.bandwidth} MHz")
        print(f"Frequency of the channel: {self.freq}")
        print("=============================================================================")


    def create_graph_data_bm(self, num_user, is_train = False):
        num_ue = num_user
        if is_train:
            num_sample = self.num_train
            data_list = self.proc_data(num_sample, num_ue)
            data_loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True, num_workers=1)
            return data_loader
        else:
            num_sample = self.num_test
            data_list = self.proc_data(num_sample, num_ue)
            data_loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True, num_workers=1)
            return data_loader

    def create_graph_data_train(self, id):
        num_sample = self.num_train
        num_ue = self.num_ue_array[id]
        data_client = self.proc_data(num_sample, num_ue)
        data_loader = DataLoader(data_client, batch_size = self.batch_size, shuffle = True, num_workers = 1)
        return data_loader

    def create_graph_data_test(self, num_user):
        num_ue = num_user
        num_sample = self.num_test
        data_client = self.proc_data(num_sample, num_ue)
        data_loader = DataLoader(data_client, batch_size=self.batch_size, shuffle=False, num_workers=1)
        return data_loader

    '''
        Old scenario: 2 cells (2 clients) and the number of UEs in each client is randomly chosen from 10 to 60.
        New scenario: the number of cells and number of UEs in each cell is fixed from the initial stage.
            10 clients, the number of UEs in each clients is [10,20,30,40,50,60,70,80,90,100] respectively.
            Each client create its own training dataset (include #num_train samples, 1 graph in each sample has corresponding #num_ue nodes)
            --> After training stage, each client has own train_loss value respectively.
                - train_loss_client0                - train_loss_client 5
                - train_loss_client1                - train_loss_client 6
                - train_loss_client2                - train_loss_client 7
                - train_loss_client3                - train_loss_client 8
                - train_loss_client4                - train_loss_client 9
            --> These loss values will be back-propagated to update local model parameter, we have
                - omega_0                           - omega_5
                - omega_1                           - omega_6
                - omega_2                           - omega_7
                - omega_3                           - omega_8
                - omega_4                           - omega_9  
            10 omega_sets will be sent to the server and then being avg process, new global omega: omega_avg
            this omega_avg will test with 3 different testing dataset: testdata_10, testdata_50, testdata_100.
            --> We will have 3 testing loss value respectively: (and on the right column is the bench mark to compare)
                - test_loss_10                      - centralized_train10_test10 / centralized_train50_test10 / centralized_train100_test10 
                - test_loss_50                      - centralized_train50_test_10 / centralized_train50_test_50 / centralized_train50_test_100
                - test_loss_100                     - centralized_train100_test_10 / centralized_train100_test_50 / centralized_train100_test_100
            We hope that with the same number of epochs (in FL case, local iteration =1), FL_GNN has approximate performance with centralized GNN case
    '''

    def proc_data(self, num_sample, num_ue):
        data_list = []
        X, Y, A, Y2 = self.generate_wGaussian(num_ue, num_sample)
        cg = self.get_cg(num_ue)
        for sample in range(num_sample):
            data = self.build_graph(X[sample], A[sample], cg, num_ue)
            data_list.append(data)
        return data_list

    def generate_wGaussian(self, num_ue, num_sample):
        Pmax = 1
        Pini = Pmax * np.ones((num_sample, num_ue, 1))
        alpha = np.random.rand(num_sample, num_ue)
        alpha2 = np.ones((num_sample, num_ue))
        CH = self.create_pathloss(num_ue, num_sample)
        H = abs(CH)
        Y = self.batch_WMMSE(Pini, alpha, H, Pmax)
        Y2 = self.batch_WMMSE(Pini, alpha2, H, Pmax)
        return H, Y, alpha, Y2

    def create_pathloss(self, num_ue, num_sample):
        pathloss_client = np.zeros((num_sample, num_ue, num_ue))
        for ite in range(num_sample):
            AP = np.random.uniform(0, 0, size = (num_ue, 2))
            UE = np.random.uniform(-1,1, size = (num_ue, 2))
            pathloss_sample = np.zeros((num_ue, num_ue))
            for m in range(num_ue):
                for k in range(num_ue):
                    dist = np.linalg.norm(AP[m] - UE[k, :])
                    if dist < self.d0:
                        PL = -self.L - 35 * np.log10(self.d1) + 20 * np.log10(self.d1) - 20 * np.log10(self.d0)
                    elif dist >= self.d0 and dist <= self.d1:
                        PL = -self.L - 35 * np.log10(self.d1) + 20 * np.log10(self.d1) - 20 * np.log10(dist)
                    else:
                        PL = -self.L - 35 * np.log10(dist) + np.random.normal(0,1) * 7

                    pathloss_sample[m, k] = 10 ** (PL / 10) * self.Pd
            pathloss_client[ite,:,:] = pathloss_sample
        return pathloss_client

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

    def show_result_graph(self):
        seed = 2017
        for i in range(self.num_clients):
            num_users = self.num_ue_array[i]
            num_tests = self.num_test
            Xtest, Ytest, Atest, Ytest2 = self.generate_wGaussian(num_users, num_tests)
            baseline_Y = self.simple_greedy(Xtest, Atest, Ytest, num_users)
            self.greedy.append(self.np_sum_rate(Xtest, baseline_Y, Atest))
            self.weighted_case.append(self.np_sum_rate(Xtest.transpose(0, 2, 1), Ytest, Atest))
            self.unweighted_case.append(self.np_sum_rate(Xtest.transpose(0, 2, 1), Ytest2, Atest))
        print(f'Greedy - Baseline Methods:  {sum(self.greedy)/self.num_clients}')
        print(f'WMMSE - Weighted Case:      {sum(self.weighted_case)/self.num_clients}')
        print(f'WMMSE - Unweighted Case:    {sum(self.unweighted_case)/self.num_clients}')


    def calculate_optimization(self, case = 1):
        if case == 1:
            num_users = 10
        elif case == 2:
            num_users = 50
        else:
            num_users = 100
        num_tests = self.num_test
        Xtest, Ytest, Atest, Ytest2 = self.generate_wGaussian(num_users, num_tests)
        optimization = self.np_sum_rate(Xtest.transpose(0, 2, 1), Ytest, Atest)
        return optimization




