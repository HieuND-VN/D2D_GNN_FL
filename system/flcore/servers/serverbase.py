import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import matplotlib.pyplot as plt


class Server(object):
    def __init__(self, args, times, env):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc        = []
        self.rs_test_loss_client   = []
        self.rs_test_loss_10       = []
        self.rs_test_loss_50       = []
        self.rs_test_loss_100      = []
        self.rs_test_auc        = []
        self.rs_train_loss      = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.train_loss_save = []
        self.test_loss_client_save = []
        self.test_loss_10_save = []
        self.test_loss_50_save = []
        self.test_loss_100_save = []


    def set_clients(self, args, clientObj, env):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):

            client = clientObj(args,
                               env,
                               id=i,
                               train_samples = args.num_train,
                               test_samples = args.num_test,
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)


    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = self.clients

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)  # Gán parameter tu global ve tung local

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(self.selected_clients, int((1 - self.client_drop_rate) * self.join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.num_train
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.num_train)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        a = 0
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        loss_test = []
        loss_test_10 = []
        loss_test_50 = []
        loss_test_100 = []
        for c in self.clients:
            loss, loss_10, loss_50, loss_100 = c.test_metrics()
            loss_test.append(loss)
            loss_test_10.append(loss_10)
            loss_test_50.append(loss_50)
            loss_test_100.append(loss_100)
        ids = [c.id for c in self.clients]
        return ids, loss_test, loss_test_10, loss_test_50, loss_test_100


    def train_metrics(self):
        loss_train = []
        for c in self.clients:
            loss = c.train_metrics()
            loss_train.append(loss)
        ids = [c.id for c in self.clients]
        return ids, loss_train


    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats_test = self.test_metrics()
        stats_train = self.train_metrics()
        test_loss_client = sum(stats_test[1]) / int(self.num_clients)
        test_loss_10 = sum(stats_test[2]) / int(self.num_clients)
        test_loss_50 = sum(stats_test[3]) / int(self.num_clients)
        test_loss_100 = sum(stats_test[4]) / int(self.num_clients)
        train_loss = sum(stats_train[1]) / int(self.num_clients)
        # test_loss_client = sum(stats_test[1])
        # test_loss_10 = sum(stats_test[2])
        # test_loss_50 = sum(stats_test[3])
        # test_loss_100 = sum(stats_test[4])
        # train_loss = sum(stats_train[1])
        if acc == None:
            self.rs_test_loss_client.append(test_loss_client)
            self.rs_test_loss_10.append(test_loss_10)
            self.rs_test_loss_50.append(test_loss_50)
            self.rs_test_loss_100.append(test_loss_100)
        else:
            acc.append(test_loss_client)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        count = 0
        for parameter in self.global_model.parameters():
            count += len(parameter.data)

            # print(f'The total number of parameters in all clients: {count*self.num_clients}')
        print("Averaged Train Loss: {:.4f}".format(train_loss*-1))
        print("Averaged Test Loss LOCAL: {:.4f}".format(test_loss_client*-1))
        print("Averaged Test Loss 10: {:.4f}".format(test_loss_10 * -1))
        print("Averaged Test Loss 50: {:.4f}".format(test_loss_50 * -1))
        print("Averaged Test Loss 100: {:.4f}".format(test_loss_100 * -1))
        self.train_loss_save.append(train_loss*-1)
        self.test_loss_client_save.append(test_loss_client*-1)
        self.test_loss_10_save.append(test_loss_10 * -1)
        self.test_loss_50_save.append(test_loss_50 * -1)
        self.test_loss_100_save.append(test_loss_100 * -1)

    # def print_(self, test_acc, test_auc, train_loss):
    #     print("Average Test Accuracy: {:.4f}".format(test_acc))
    #     print("Average Test AUC: {:.4f}".format(test_auc))
    #     print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True


    def illustrate(self, env):
        x = np.arange(0, self.global_rounds+1)
        optimization = np.full_like(x,1)*(sum(env.weighted_case)/self.num_clients)
        # SumRateMMSEplot = np.full_like(x,1)*7.5
        plt.plot(self.train_loss_save, label = 'Training')
        plt.plot(self.test_loss_client_save, label='Testing N = local number')
        plt.plot(self.test_loss_10_save, label='Testing N = 10')
        plt.plot(self.test_loss_50_save, label='Testing N = 50')
        plt.plot(self.test_loss_100_save, label='Testing N = 100')
        plt.plot(x,optimization, label = 'Optimization')
        plt.xlabel('Number of epoch')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('Reward_global_vs_test.png', bbox_inches='tight')
        plt.show()

    def illustrate_bm(self, env, loss_tr10, loss_tr50, loss_tr100, case = 1):
        x = np.arange(0, self.global_rounds+1)
        optimization = np.full_like(x,1)*(env.calculate_optimization(case))
        plt.plot(self.test_loss_client_save, label='GraphFL global test')
        if case == 1:
            plt.plot(self.test_loss_10_save, label='Decentralized, Te = 10')
            plt.plot(loss_tr10, label='Centralized , Tr = 10, Te = 10')
            plt.plot(loss_tr50, label='Centralized , Tr = 50, Te = 10')
            plt.plot(loss_tr100, label='Centralized , Tr = 100, Te = 10')
            plt.plot(x, optimization, label='Optimization')
            plt.xlabel('Number of epoch')
            plt.ylabel('Reward')
            plt.savefig('Reward_test10.png', bbox_inches='tight')
            plt.show()
        elif case == 2:
            plt.plot(self.test_loss_50_save, label='Decentralized, Te = 50')
            plt.plot(loss_tr10, label='Centralized , Tr = 10, Te = 50')
            plt.plot(loss_tr50, label='Centralized , Tr = 50, Te = 50')
            plt.plot(loss_tr100, label='Centralized , Tr = 100, Te = 50')
            plt.plot(x, optimization, label='Optimization')
            plt.xlabel('Number of epoch')
            plt.ylabel('Reward')
            plt.savefig('Reward_test50.png', bbox_inches='tight')
            plt.show()
        else:
            plt.plot(self.test_loss_100_save, label='Decentralized, Te = 100')
            plt.plot(loss_tr10, label='Centralized , Tr = 10, Te = 100')
            plt.plot(loss_tr50, label='Centralized , Tr = 50, Te = 100')
            plt.plot(loss_tr100, label='Centralized , Tr = 100, Te = 100')
            plt.plot(x, optimization, label='Optimization')
            plt.xlabel('Number of epoch')
            plt.ylabel('Reward')
            plt.savefig('Reward_test100.png', bbox_inches='tight')
            plt.show()

