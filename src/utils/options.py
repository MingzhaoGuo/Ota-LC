import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--frac', type=float, default= 0.5, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=100, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset, cifar or mnist")
    parser.add_argument('--iid', default=1, action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1)')
    parser.add_argument('--lr-scheduler', type = int,default=1, help='whether using lr scheduler')

    # arguments for compression methods
    parser.add_argument('--mode', type=str, default='ota_cs', help="sgd, ota_lc, blue_cs, ota_rlc or ota_cs")
    parser.add_argument('--C', type=float, default= 0.2, help="compression ratio")

    # arguments for ota_cs and blue_cs
    parser.add_argument('--iter_cs', type = int, default=5, help="iterations for signal recovery in turbo cs")

    # arguments for MIMO Wireless
    parser.add_argument('--dimension', type = int, default= 8)
    parser.add_argument('--Ns', type=int, default=8)
    parser.add_argument('--Nt', type = int, default= 8)
    parser.add_argument('--Nr', type = int, default= 80)
    parser.add_argument('--SNRdB', type=int, default= 20)
    

    args = parser.parse_args()
    return args
