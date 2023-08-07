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
        for client in self.clients:
            print(f'Number of mobile user in client[{client.id}]: [{client.num_ue_case1}]')


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
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

        print(f'Best result: -Train: {min(self.train_loss_save)} -Test: {min(self.test_loss_save)}')

        self.illustrate()


        print("\nAverage time cost per round.", sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
