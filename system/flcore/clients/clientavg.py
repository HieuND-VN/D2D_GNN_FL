import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import initialize_dp, get_dp_params


class clientAVG(Client):
    def __init__(self, args, env,  id, train_samples, test_samples, **kwargs):
        super().__init__(args, env, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.eval()
        self.model.train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer,
                                                                                    trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, data in enumerate(trainloader):
                if type(data) == type([]):
                    data[0] = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.sr_loss(data, output, self.num_ue_case1)
                loss.backward()
                self.optimizer.step()

                if self.learning_rate_decay:
                   self.learning_rate_scheduler.step()

                self.train_time_cost['num_rounds'] += 1
                self.train_time_cost['total_cost'] += time.time() - start_time

                if self.privacy:
                    eps, DELTA = get_dp_params(privacy_engine)
                    print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")