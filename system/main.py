import torch
import argparse
import os
import time
import warnings
import numpy as np
import logging

from flcore.servers.serveravg import FedAvg
from flcore.benchmark.benchmark import *

from flcore.trainmodel.models import *

from utils.mem_utils import MemReporter

from utils.environment import Env

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)
def run(args, env, benchmark_10, benchmark_50, benchmark_100):
    time_list = []
    reporter = MemReporter()
    env.show_result_graph()
    #Calculate to compare with 1 cell train and test with same number of UE
    loss_tr10, loss_tr10_te10, loss_tr10_te50, loss_tr10_te100 = benchmark_10.calculate()
    loss_tr50, loss_tr50_te10, loss_tr50_te50, loss_tr50_te100 = benchmark_10.calculate()
    loss_tr100, loss_tr100_te10, loss_tr100_te50, loss_tr100_te100 = benchmark_10.calculate()


    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        args.model = IGCNet().to(args.device)
        print(args.model)
        server = FedAvg(args, i, env)
        server.train(env)
        time_list.append(time.time() - start)
        server.illustrate(env)

    print(f"\n>>>>>>>>>>>>Average time cost: {round(np.average(time_list), 2)}s.")
    # env.show_result_graph()
    # Global average
    # show_result_graph(args.num_user, args.test_samples, args.var_db, args.num_clients)

    reporter.report()
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



if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('-go', "--goal", type=str, default="test", help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="CFMIMO")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=50)
    parser.add_argument('-ls', "--local_steps", type=int, default=1)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # GraphNN
    parser.add_argument('-ntr', "--num_train", type = int, default = 3000)
    parser.add_argument('-nte', "--num_test", type = int, default = 500)
    parser.add_argument('-uemin', "--num_ue_min", type = int, default = 10)
    parser.add_argument('-uemax', "--num_ue_max", type = int, default = 100)
    parser.add_argument('-var', "--var_db", type = int, default = 1)
    parser.add_argument('-itf', "--interference", type = float, default = 0.5,
                        help="Normalize Interference in Case2")
    parser.add_argument('-stepsz', "--step_size", type = int, default = 20,
                        help="Step size using in scheduler_clientbase")
    # D2D scenario
    parser.add_argument('-bw', "--bandwidth", type = int, default = 20, help = "Bandwidth in MHz")
    parser.add_argument('-d', "--diameter", type = float, default = 1, help = "Diameter of region in Km")
    parser.add_argument('-ha', "--height_ap", type = float, default = 15, help = 'Height of Access Point')
    parser.add_argument('-hm', "--height_mobile", type = float, default = 1.65, help = "Height of mobile user")
    parser.add_argument('-f', "--frequency", type=float, default=1900, help="Frequency in MHz")
    parser.add_argument('-pf', "--power_f", type=float, default=0.2, help="Downlink Power")
    parser.add_argument('-thnoise', "--ther_noise", type=float, default=20000000 * 10 ** (-17.4) * 10 ** -3)
    parser.add_argument('-d0', "--distance_0", type=float, default=0.01, help="distance_0 in km")
    parser.add_argument('-d1', "--distance_1", type=float, default=0.05, help="distance_1 in km")

    parser.add_argument('-n', "--N", type=int, default=50)

    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 60)

    print("Algorithm: \t{}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learning rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threshold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Local model: Graph NN")
    print("Using device: {}".format(args.device))
    print("Dataset: D2D Generate")
    print("===========================Graph_Infor======================================")
    print(f"Number of user equipment range: [{args.num_ue_min}; {args.num_ue_max}]")
    print("Number of training samples (each client): {}".format(args.num_train))
    print("Number of testing samples (each client): {}".format(args.num_test))
    print("============================================================================")

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)
    env = Env(args)
    env.env_print()
    trainloader_bm10 = env.create_graph_data_bm(num_user=10, is_train=True)
    trainloader_bm50 = env.create_graph_data_bm(num_user=50, is_train=True)
    trainloader_bm100 = env.create_graph_data_bm(num_user=100, is_train=True)
    testloader_bm10 = env.create_graph_data_bm(num_user=10, is_train=False)
    testloader_bm50 = env.create_graph_data_bm(num_user=50, is_train=False)
    testloader_bm100 = env.create_graph_data_bm(num_user=100, is_train=False)
    benchmark_10 = Benchmark10(args, trainloader_bm10, testloader_bm10, testloader_bm50, testloader_bm100)
    benchmark_50 = Benchmark50(args, trainloader_bm50, testloader_bm10, testloader_bm50, testloader_bm100)
    benchmark_100 = Benchmark100(args, trainloader_bm100, testloader_bm10, testloader_bm50, testloader_bm100)
    run(args, env, benchmark_10, benchmark_50, benchmark_100)




