import torch
from main import test
from CWRUDataset import CWRUDataset
from torch.utils.data import DataLoader
from DSAN import DSAN
C = {'OR007':'dataset/CWRU/12k_Drive_End_OR007@6_2_132.mat',
    'OR021':'dataset/CWRU/12k_Drive_End_OR021@6_2_236.mat',
    'IR021':'dataset/CWRU/12k_Drive_End_IR021_2_211.mat',
    'B007':'dataset/CWRU/12k_Drive_End_B007_2_120.mat',
    'B021':'dataset/CWRU/12k_Drive_End_B021_2_224.mat',
    'OR014':'dataset/CWRU/12k_Drive_End_OR014@6_2_199.mat',
    'IR014':'dataset/CWRU/12k_Drive_End_IR014_2_171.mat',
    'IR007':'dataset/CWRU/12k_Drive_End_IR007_2_107.mat',
    'B014':'dataset/CWRU/12k_Drive_End_B014_2_187.mat'}
dataset = CWRUDataset(C)
loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)
model = torch.load("model.pkl")
test(model,loader)