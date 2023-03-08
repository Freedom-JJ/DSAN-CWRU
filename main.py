import torch
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os
import torchvision.transforms as transforms
from DSAN import DSAN
from data_loader import GetLoader
import data_loader
from torchvision import datasets
from torch.utils.data import DataLoader,ConcatDataset
from CWRUDataset import CWRUDataset
def load_data(root_path, src, tar, batch_size):
    A = {'B007':'dataset/CWRU/12k_Drive_End_B007_0_118.mat',
     'OR021':'dataset/CWRU/12k_Drive_End_OR021@6_0_234.mat',
     'OR014':'dataset/CWRU/12k_Drive_End_OR014@6_0_197.mat',
     'B014':'dataset/CWRU/12k_Drive_End_B014_0_185.mat',
     'IR021':'dataset/CWRU/12k_Drive_End_IR021_0_209.mat',
     'B021':'dataset/CWRU/12k_Drive_End_B021_0_222.mat',
     'IR007':'dataset/CWRU/12k_Drive_End_IR007_0_105.mat',
     'IR014':'dataset/CWRU/12k_Drive_End_IR014_0_169.mat',
     'OR007':'dataset/CWRU/12k_Drive_End_OR007@6_0_130.mat'}
    B = {'IR007':'dataset/CWRU/12k_Drive_End_IR007_1_106.mat',
     'IR014':'dataset/CWRU/12k_Drive_End_IR014_1_170.mat',
    'B007':'dataset/CWRU/12k_Drive_End_B007_1_119.mat',
    'B021':'dataset/CWRU/12k_Drive_End_B021_1_223.mat',
    'OR021':'dataset/CWRU/12k_Drive_End_OR021@6_1_235.mat',
    'B014':'dataset/CWRU/12k_Drive_End_B014_1_186.mat',
    'IR021':'dataset/CWRU/12k_Drive_End_IR021_1_210.mat',
    'OR007':'dataset/CWRU/12k_Drive_End_OR007@6_1_131.mat',
    'OR014':'dataset/CWRU/12k_Drive_End_OR014@6_1_198.mat'}
    C = {'OR007':'dataset/CWRU/12k_Drive_End_OR007@6_2_132.mat',
    'OR021':'dataset/CWRU/12k_Drive_End_OR021@6_2_236.mat',
    'IR021':'dataset/CWRU/12k_Drive_End_IR021_2_211.mat',
    'B007':'dataset/CWRU/12k_Drive_End_B007_2_120.mat',
    'B021':'dataset/CWRU/12k_Drive_End_B021_2_224.mat',
    'OR014':'dataset/CWRU/12k_Drive_End_OR014@6_2_199.mat',
    'IR014':'dataset/CWRU/12k_Drive_End_IR014_2_171.mat',
    'IR007':'dataset/CWRU/12k_Drive_End_IR007_2_107.mat',
    'B014':'dataset/CWRU/12k_Drive_End_B014_2_187.mat'}
    D = {'OR021':'dataset/CWRU/12k_Drive_End_OR021@6_3_237.mat',
    'B014':'dataset/CWRU/12k_Drive_End_B014_3_188.mat',
    'B007':'dataset/CWRU/12k_Drive_End_B007_3_121.mat',
    'IR014':'dataset/CWRU/12k_Drive_End_IR014_3_172.mat',
    'B021':'dataset/CWRU/12k_Drive_End_B021_3_225.mat',
    'OR014':'dataset/CWRU/12k_Drive_End_OR014@6_3_200.mat',
    'OR007':'dataset/CWRU/12k_Drive_End_OR007@6_3_133.mat',
    'IR007':'dataset/CWRU/12k_Drive_End_IR007_3_108.mat',
    'IR021':'dataset/CWRU/12k_Drive_End_IR021_3_212.mat'}
    batch_size = 32
    source_A_dataset =ConcatDataset([CWRUDataset(A),CWRUDataset(B)])
    target_D_dataset = ConcatDataset([CWRUDataset(C),CWRUDataset(D)])
    test_D_dataset = ConcatDataset([CWRUDataset(C),CWRUDataset(D)])
    source = DataLoader(dataset=source_A_dataset,batch_size=batch_size,shuffle=True)
    target = DataLoader(dataset=target_D_dataset,batch_size=batch_size,shuffle=True)
    test   = DataLoader(dataset=test_D_dataset,batch_size=batch_size,shuffle=True) 
    return source , target , test
    # kwargs = {'num_workers': 1, 'pin_memory': True}
    # loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    # loader_tar = data_loader.load_training(root_path, tar, batch_size, kwargs)
    # loader_tar_test = data_loader.load_testing(
    #     root_path, tar, batch_size, kwargs)
    # return loader_src, loader_tar, loader_tar_test


def train_epoch(epoch, model, dataloaders, optimizer):
    

    model.train()
    source_loader, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)
    print("-----num iter----:"+str(num_iter))
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)
        data_source, label_source = data_source.cuda(1), label_source.cuda(1)
        data_target = data_target.cuda(1)
        optimizer.zero_grad()
        label_source_pred, loss_lmmd = model(
            data_source, data_target, label_source)
        # onehot = torch.zeros((label_source.shape[0],args.nclass),dtype=torch.float)
        # onehot = onehot.cuda(1)
        # onehot = onehot.scatter_(dim=1 , index=label_source.unsqueeze(dim=1),src=torch.ones(label_source.shape[0],args.nclass,device=3))
        # label_source = onehot
        loss_cls = F.nll_loss(F.log_softmax( #log_softmax的值巨大，非常不合理，改成soft_max这一步骤应该写到forward中，但是也不合理，因为loss是负数
            label_source_pred, dim=1), label_source)
        # loss_cls = F.nll_loss(label_source_pred,label_source)
        # loss_cls = F.cross_entropy(F.softmax(label_source_pred,dim=1),label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1
        loss = loss_cls + args.weight * lambd * loss_lmmd

        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print(f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')


def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(1), target.cuda(1)
            pred = model.predict(data)
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataloader)
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    return correct


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='./dataset//Original_images/')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='amazon/images')
    parser.add_argument('--tar', type=str,
                        help='Target domain', default='webcam/images')
    parser.add_argument('--nclass', type=int,
                        help='Number of classes', default=9)
    parser.add_argument('--batch_size', type=float,
                        help='batch size', default=32)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=200)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001, 0.01, 0.01])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=30)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2021)
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='1')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataloaders = load_data(args.root_path, args.src,
                            args.tar, args.batch_size)
    model = DSAN(num_classes=args.nclass).cuda(1)
    # model = torch.nn.DataParallel(model)
    correct = 0
    stop = 0

    if args.bottleneck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': args.lr[1]},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[2]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)

    for epoch in range(1, args.nepoch + 1):
        stop += 1
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = args.lr[index] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)

        train_epoch(epoch, model, dataloaders, optimizer)
        s_correct = test(model,dataloaders[0])
        print("source accuary:" +str( 100. * s_correct / len(dataloaders[0].dataset)))
        t_correct = test(model, dataloaders[-1])
        if t_correct > correct:
            correct = t_correct
            stop = 0
            torch.save(model, 'model.pkl')
            print("------ save ----")
        print(
            f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')

        if stop >= args.early_stop:
            print(
                f'Final test acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
            break
