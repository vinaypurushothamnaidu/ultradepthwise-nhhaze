import torch.utils.data as data
import torchvision.transforms as tfs
import os, sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt

BS = opt.bs

def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, format='.png'):
        super(RESIDE_Dataset,self).__init__()
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'hazy'))
        self.haze_imgs=[os.path.join(path,'hazy',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'clear')

    def __getitem__(self, index):
        dim = (512, 512)
        haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        id = img.split('/')[-1].split('_')[0]
        clear_name = id+self.format
        clear = Image.open(os.path.join(self.clear_dir,clear_name))
        haze = haze.resize(dim)
        clear = clear.resize(dim)
        haze = haze.convert("RGB")
        clear = clear.convert("RGB") 
        haze = tfs.ToTensor()(haze)
        clear = tfs.ToTensor()(clear)
        return haze, clear

    def __len__(self):
        return len(self.haze_imgs)

# path = '/Users/vinaynaidu/Desktop'

# NH_haze_train_loader = DataLoader(dataset = RESIDE_Dataset(path + '/NH-HAZE/NH-HAZE', train = True), 
#                              batch_size = BS, shuffle = True)

# NH_haze_test_loader = DataLoader(dataset = RESIDE_Dataset(path + '/NH-HAZE/NH-HAZE-VAL', train = False), 
#                              batch_size = 1, shuffle = False)


path = '/kaggle/input/'

NH_haze_train_loader = DataLoader(dataset = RESIDE_Dataset(path + 'nh-haze/NH-HAZE/train', train = True), 
                             batch_size = BS, shuffle = True)

NH_haze_test_loader = DataLoader(dataset = RESIDE_Dataset(path + 'nh-haze/NH-HAZE/val', train = False), 
                             batch_size = 1, shuffle = False)



if __name__ == "__main__":
    pass