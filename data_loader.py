from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import glob
import numpy as np
from scipy.fftpack import fft,fftshift
from scipy.signal import stft
def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader







class GetLoader(Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data
class Get1DLoader(Dataset):
    def __init__(self,globpath,path = None,trans = None):
        if path is not None:
            self.pathlist = path
        else:
            self.pathlist = glob.glob(globpath)
        self.trans = trans
        self.index = 0
        self.datalist = readfile(self.pathlist[self.index])
        self.rawdata = [readfile(i) for i in self.pathlist]
        
        self.len = 1000 * len(self.pathlist)
        self.randomclass = np.random.randint(0,self.len/1000,self.len)
        self.randomstart = np.random.randint(0,1000,self.len)
    def __getitem__(self,item):
        # if item % 10 ==0 and item != 0:
        #     self.reload()
        # count = item % 10
        # stop = len(self.datalist) - 20000
        # datarange = np.linspace(0,stop,10,dtype=np.int32)
        # label =  self.pathlist[self.index].split("\\")[-3]
        # label = int(label)
        # pointstart = datarange[count]
        # pointend = datarange[count] + 20000
        #### 简单随机化
        randindex = self.randomclass[item]
        rawdata = self.rawdata[randindex]
        randstart = self.randomstart[item]
        stop = len(rawdata) - 20000
        datarange = np.linspace(0,stop,1000,dtype=np.int32)
        pointstart = datarange[randstart]
        pointend = pointstart + 20000
        label =  self.pathlist[randindex].split("/")[-3]
        label = int(label)
        ####
        
        
        
        _,_,data_stft = stft(np.array(rawdata[pointstart:pointend]),fs=20000)
        data_stft = np.abs(data_stft)
        data_stft = np.array(data_stft,dtype=np.float32)
        data_stft = data_stft[:128,:128]
        norms = np.linalg.norm(data_stft,axis=1)
        return data_stft/norms , label    #必须要随机化

    
    def __len__(self):
        return 1000* len(self.pathlist)
    
    def reload(self):
        # print("reload")
        # self.count = 0
        self.index = self.index + 1
        self.datalist = readfile(self.pathlist[self.index])


def readfile(filename):
    dataNum = 0
    max = 0
    dataList = []
    with open(filename,"r") as f:
         for line in f.readlines():
                linestr = line.strip('\n')
                dataList.append(float(linestr))
                dataNum += 1
                if float(linestr) >= max:
                    max = float(linestr)
    return dataList
def spectrum(signal:np.ndarray):
    N = 20000                        # 采样点数
    sample_freq=20000                 # 采样频率 120 Hz, 大于两倍的最高频率
    sample_interval=1/sample_freq   # 采样间隔
    signal_len=N*sample_interval    # 信号长度
    t=np.arange(0,signal_len,sample_interval)
    fft_data = fft(signal)
    # 这里幅值要进行一定的处理，才能得到与真实的信号幅值相对应
    fft_amp0 = np.array(np.abs(fft_data)/N*2)   # 用于计算双边谱
    direct=fft_amp0[0]
    fft_amp0[0]=0
    N_2 = int(N/2)
    fft_amp1 = fft_amp0[0:N_2]  # 单边谱
    # 计算频谱的频率轴
    list1 = np.array(range(0, int(N/2)))
    freq1 = sample_freq*list1/N        # 单边谱的频率轴
    return (fft_amp1,freq1)
def feature_extra(timeDomainData,sampleRate=20000):
    reslut={}
    N = len(timeDomainData) #采样个数
    # 时域特征
    F1 = sum(timeDomainData) / N #均值
    F2 = (sum((timeDomainData - F1)**2)/(N-1))**0.5 #标准差
    F3 = (np.sum(np.power(np.abs(timeDomainData),0.5))/N)**2 #F3与原公式有点不一样，原公式不求绝对值，但是信号有负数
    F4 = (np.sum(np.power(timeDomainData,2))/N)**0.5
    F5 = np.max(np.abs(timeDomainData))                     #最大值
    F6 = np.sum(np.power(timeDomainData-F1,3))/((N-1)*F2**3)  #偏度系数 可以描述分布的形状特征，刻画分布对称性
    F7 = np.sum((timeDomainData-F1)**4)/((N-1)*(F2**4))       #峰度系数，可以描述分布的形状特征，刻画分布的陡峭性
    F8 = F5 /F4
    F9 =F5/F3
    F10 =F4/(1/N*np.sum(np.abs(timeDomainData)))
    F11 = F10/F4 * F5
    #计算单边谱,因为双边谱完全是对称的
    fft_data = fft(timeDomainData)
    fft_amp = np.array(np.abs(fft_data)/N*2)[0:int(N/2)] #s(k)
    fft_amp[0] *= 0.5
    #绘制频率轴
    list = np.array(range(0, int(N/2)))
    freq = sampleRate*list/N #f(k)
    #频域特征
    K = list.size   #谱线数
    F12 = np.sum(fft_amp)/K
    F13 = np.sum((fft_amp-F12)**2)/(K-1)
    F14 = np.sum((fft_amp-F12)**3)/(K*(F13**2))
    F15 = np.sum((fft_amp-F12)**4)/(K*(F13**2))
    F16 = np.sum((fft_amp*freq))/(F12*K)
    F17 = (np.sum((freq - F16)**2 * fft_amp)/K)**0.5
    temp = np.sum(freq**2 * fft_amp)
    F18 = (temp/(F12*K))**0.5
    F19 = (np.sum(freq**4 * fft_amp)/temp)**0.5
    F20 = temp /((F12*K)**0.5 * F19*temp**0.5)
    F21 = F17 / F16
    F22 = np.sum((freq-F16)**3 * fft_amp)/(K*F17**3)
    F23 = np.sum((freq-F16)**4 * fft_amp)/(K * F17**4)
    # F24 = np.sum((freq - F16)**0.5 * fft_amp)/(K* F17**0.5) #开方也出现错误
    #简单统计值
    efficient = np.sum(fft_amp)
    rms = np.mean(fft_amp**2)**0.5
    reslut['timeDomain']=[F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11]
    reslut['frequencyDomain'] = [F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F23]
    reslut['simple'] = [efficient,rms]
    return reslut
