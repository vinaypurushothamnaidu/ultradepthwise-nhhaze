import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import os
import time
import network
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch 
from tqdm import tqdm
from torchvision.utils import save_image
import network

my_model = network.B_transformer()
my_model=nn.DataParallel(my_model)
checkpoint = torch.load('./its_train_ultranh.pk', map_location=torch.device('cpu'))
my_model.load_state_dict(checkpoint['model']) 
my_model.eval()

to_pil_image = transforms.ToPILImage()

tfs_full = transforms.Compose([
            #transforms.Resize(1080),
            transforms.ToTensor()
        ])

image_in = Image.open('./01_outdoor_hazy.jpg').convert('RGB')
full = tfs_full(image_in).unsqueeze(0)
output = my_model(full)
save_image(output[0], '01_outdoor_clear.jpg')