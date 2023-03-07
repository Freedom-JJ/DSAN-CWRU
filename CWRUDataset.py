from torch.utils.data import Dataset,DataLoader
from scipy.signal import stft
import numpy as np
import scipy.io as sio
class CWRUDataset(Dataset):
    def __init__(self,pathlist) -> None:
        super().__init__()
        self.pathdict = pathlist
        self.label = ['B007','B014','B021','IR007','IR014','IR021','OR007','OR014','OR021']
        self.datalist = [self.loadFile(i) for i in self.label]
        self.startrange = np.linspace(0,100000,200,dtype=np.int32) 
    def __len__(self):
        return 1000
    def __getitem__(self, index) :
        i = index % len(self.label)
        slice = index // len(self.label)
        start = self.startrange[slice]
        data = self.datalist[i][start:start+20000]
        stftdata = np.abs(stft(data,12000)[2])
        return stftdata[:128,:128] , i
    def loadFile(self,label):
        file = self.pathdict[label]
        key ='X'+ file.split("_")[-1].split(".")[0] +'_DE_time'
        data = sio.loadmat(file)[key].squeeze()
        return np.array(data,dtype=np.float32)