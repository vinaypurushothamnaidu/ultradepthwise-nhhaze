import torch
import torch.optim
import network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

my_model = network.B_transformer().to(device)
my_model.eval()
my_model.to(device)
checkpoint = torch.load('./nh_train_ultrad.pk', map_location = device)

print("Max PSNR :", checkpoint['max_psnr'])
print("Max SSIM :", checkpoint['max_ssim'])
print("Trained for :", checkpoint['step'])