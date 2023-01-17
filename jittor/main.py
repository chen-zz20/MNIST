import argparse
import os
from tqdm import tqdm
import time

from tensorboardX import SummaryWriter

from dataload import Train_Data, Test_Data
from model import Model, train_epoch, test_epoch

import jittor as jt
import jittor.nn as nn
import jittor.optim as optim
from jittor.dataset.dataset import DataLoader

if jt.compiler.has_cuda:
    jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100,
	help='Batch size for mini-batch training and evaluating. Default: 100')
parser.add_argument('--num_epochs', type=int, default=10,
	help='Number of training epoch. Default: 100')
parser.add_argument('--hidden_dim', type=int, default=128,
	help='Number of hidden dim. Default: 128')
parser.add_argument('--learning_rate', type=float, default=1e-3,
	help='Learning rate during optimization. Default: 1e-3')
parser.add_argument('--test', default=False, action="store_true", 
	help='True to train and False to inference. Default: True')
parser.add_argument('--data_dir', type=str, default='../data',
	help='Data directory. Default: ../data')
parser.add_argument('--train_dir', type=str, default='./train',
	help='Training directory for saving model. Default: ./train')
parser.add_argument('--log_dir', type=str, 
default='./log', 
    help='The path of the log directory')
parser.add_argument('--name', type=str, default='test',
	help='Give the model a name. Default: test')
parser.add_argument('--pretrained', type=str, default=None,
	help='Give a pretrained model. Default: None')
args = parser.parse_args()


if __name__ == '__main__':
    train_loader = DataLoader(Train_Data(args.data_dir), batch_size=args.batch_size, shuffle=True, num_workers=10)
    test_loader = DataLoader(Test_Data(args.data_dir), batch_size=args.batch_size, num_workers=10)
    model = Model(784, args.hidden_dim, 10)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    log_dir = os.path.join(args.log_dir, 'train')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    train_writer = SummaryWriter(log_dir=log_dir)

    log_dir = os.path.join(args.log_dir, 'test')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    test_writer = SummaryWriter(log_dir=log_dir)

    if args.pretrained is not None:
        model_path = os.path.join(args.train_dir, f"model-{args.pretrained}.pth.tar")
        if os.path.exists(model_path):
            model = jt.load(model_path)
    loss_metrics = nn.MSELoss()

    if not args.test:
        print("begin trainning")
        begin = time.time()
        for epoch in tqdm(range(1, args.num_epochs+1)):
            train_loss , train_acc = train_epoch(model, train_loader, loss_metrics, optimizer)
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("accuracy", train_acc, epoch)

            test_loss, test_acc = test_epoch(model, test_loader, loss_metrics)
            test_writer.add_scalar("loss", test_loss, epoch)
            test_writer.add_scalar("accuracy", test_acc, epoch)
        end = time.time()
        use_time = end - begin
        minutes = use_time // 60
        use_time -= minutes * 60
        use_time = "%.3f" % (use_time)
        print("end trainning")
        print(f"{args.name} training used {minutes} minutes {use_time} seconds.")
        print("begin testing")
        test_loss, test_acc = test_epoch(model, test_loader, loss_metrics)
        print("The final loss is %.3f, final accuracy is %.3f" % (test_loss, test_acc))
        with open(os.path.join(args.train_dir, f"model-{args.name}.pth.tar"), 'wb') as fout:
            jt.save(model, fout)
        train_writer.close()
        test_writer.close()
        
    else:
        print("begin testing")
        test_loss, test_acc = test_epoch(model, test_loader, loss_metrics)
        print("The final loss is %.3f, final accuracy is %.3f" % (test_loss, test_acc))
        
