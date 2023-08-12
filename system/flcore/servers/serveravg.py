import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class FedAvg(Server):
    def __init__(self, args, times, env):
        super().__init__(args, times, env)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG, env)


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self,env):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("Evaluate global model")
                self.evaluate()
            for m, client in enumerate(self.selected_clients):
                client.train()


            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('>'*10, 'time cost: ', self.Budget[-1])

        # print(f'Best result: -Train: {min(self.train_loss_save)} -Test: {min(self.test_loss_save)}')
        '''
        Compare scenarios
        1. FL train_loss and test loss (test_loss_10, test_loss_50, test_loss_100)
        2. GNN case: 
            2.1. train_10 and test_10, test_50, test_100
            2.1. train_50 and test_10, test_50, test_100
            2.1. train_100 and test_10, test_50, test_100
            Compare with WMMSE optimization value
        3. FL test loss compare with GNN centralize models
            3.1. FL test_10 vs. train_10_test_10, train_50_test_10, train_100_test_10
            3.2. FL test_50 vs. train_10_test_50, train_50_test_50, train_100_test_50
            3.3. FL test_100 vs train_10_test_100, train_50_test_100, train_100_test_100
        '''
        self.illustrate(env)


        print("\nAverage time cost per round.", sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
