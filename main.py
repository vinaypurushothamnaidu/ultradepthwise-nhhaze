import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from data_utils import *
from torchvision.models import vgg16
import network

models_ = {
	'ultrad':network.B_transformer(),
}

loaders_ = {
	'nh_train':NH_haze_train_loader,
	'nh_test':NH_haze_test_loader
}

start_time = time.time()
T = opt.steps	

def train(net, loader_train, loader_test, optim):
	losses = []
	start_step = 0
	max_ssim = 0
	max_psnr = 0
	ssims = []
	psnrs = []
	mse = nn.L1Loss().to(opt.device)

	if opt.resume and os.path.exists(opt.model_dir):
		print(f'resume from {opt.model_dir}')
		# net = nn.DataParallel(net)
		ckp = torch.load(opt.model_dir)
		losses = ckp['losses']
		net.load_state_dict(ckp['model'])
		start_step = ckp['step']
		max_ssim = ckp['max_ssim']
		max_psnr = ckp['max_psnr']
		psnrs = ckp['psnrs']
		ssims = ckp['ssims']
		print(f'start_step:{start_step} start training ---')
	else :
		print('train from scratch *** ')

	# print(f'resume from {opt.model_dir}')
	# net = nn.DataParallel(net)
	# ckp = torch.load(str(opt.model_dir), map_location=opt.device)

	# # ckp = torch.load(opt.model_dir)

	# losses = ckp['losses']
	# net.load_state_dict(ckp['model'])
	# start_step = ckp['step']
	# max_ssim = ckp['max_ssim']
	# max_psnr = ckp['max_psnr']
	# psnrs = ckp['psnrs']
	# ssims = ckp['ssims']
	# print(f'start_step:{start_step} start training ---')

	for step in range(start_step+1, opt.steps+1):
		net.train()
		lr = opt.lr
		x,y = next(iter(loader_train))
		x = x.to(opt.device); y = y.to(opt.device)
		out = net(x)
		loss = mse(out, y)
		loss.backward()
		optim.step()
		optim.zero_grad()
		losses.append(loss.item())

		print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

		if step % opt.eval_step == 0 :
			with torch.no_grad():
				ssim_eval,psnr_eval = test(net,loader_test, max_psnr,max_ssim,step)

			print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')
			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)
			if ssim_eval > max_ssim and psnr_eval > max_psnr :
				max_ssim = max(max_ssim,ssim_eval)
				max_psnr = max(max_psnr,psnr_eval)
				torch.save({
							'step':step,
							'max_psnr':max_psnr,
							'max_ssim':max_ssim,
							'ssims':ssims,
							'psnrs':psnrs,
							'losses':losses,
							'model':net.state_dict()
				},opt.model_dir)
				print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

				np.save(f'./numpy_files/{model_name}_{step}_losses.npy', losses)
				np.save(f'./numpy_files/{model_name}_{step}_ssims.npy', ssims)
				np.save(f'./numpy_files/{model_name}_{step}_psnrs.npy', psnrs)

def test(net,loader_test,max_psnr,max_ssim,step):
	net.eval()
	torch.cuda.empty_cache()
	ssims = []
	psnrs = []
	for i ,(inputs,targets) in enumerate(loader_test):
		inputs = inputs.to(opt.device);targets=targets.to(opt.device)
		pred = net(inputs)
		ssim1 = ssim(pred,targets).item()
		psnr1 = psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
	return np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
	loader_train = loaders_[opt.trainset]
	loader_test = loaders_[opt.testset]
	net = models_[opt.net]
	net = net.to(opt.device)
	if opt.device == 'cuda':
		# net = torch.nn.DataParallel(net)
		cudnn.benchmark = True
	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	train(net, loader_train, loader_test, optimizer)
	
