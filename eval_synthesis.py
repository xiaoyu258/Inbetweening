import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
import numpy as np
import os

import models


parser = argparse.ArgumentParser()
parser.add_argument("--img0Path", type=str, default='./images/f_032242.png', help='input image0 path')
parser.add_argument("--img1Path", type=str, default='./images/f_032246.png', help='input image1 path')
parser.add_argument("--sketPath", type=str, default='./images/f_032244_sket.png', help='input sketch_t path')
parser.add_argument("--saveImgPath", type=str, default='./images/f_032244_It.png', help='saved image It path')
parser.add_argument("--modelPath", type=str, default='./checkpoints/frame_synthesis_model.ckpt', help='checkpoint path')
args = parser.parse_args()
    
torch.set_grad_enabled(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
multiGPUs = [0]

netT   = models.ResNet()
sketExt = models.PWCExtractor()
imagExt = models.PWCExtractor()
flowEst = models.Network()
blenEst = models.blendNet()

W = 576
H = 384
flowBackWarp = models.backWarp(W, H)
occlusiCheck = models.occlusionCheck(W, H)

if torch.cuda.device_count() >= 1:
    netT = nn.DataParallel(netT, device_ids=multiGPUs)
    sketExt = nn.DataParallel(sketExt, device_ids=multiGPUs)
    imagExt = nn.DataParallel(imagExt, device_ids=multiGPUs)
    flowEst = nn.DataParallel(flowEst, device_ids=multiGPUs)
    blenEst = nn.DataParallel(blenEst, device_ids=multiGPUs)
    
    flowBackWarp = nn.DataParallel(flowBackWarp, device_ids=multiGPUs)
    occlusiCheck = nn.DataParallel(occlusiCheck, device_ids=multiGPUs)

netT = netT.to(device)
sketExt = sketExt.to(device)
imagExt = imagExt.to(device)
flowEst = flowEst.to(device)
blenEst = blenEst.to(device)

flowBackWarp = flowBackWarp.to(device)
occlusiCheck = occlusiCheck.to(device)

model_dict = torch.load(args.modelPath)
for key, value in model_dict.items():
    print(key)
    
netT.load_state_dict(model_dict['state_dict_netT'])
sketExt.load_state_dict(model_dict['state_dict_sketExt'])
imagExt.load_state_dict(model_dict['state_dict_imagExt'])
flowEst.load_state_dict(model_dict['state_dict_flowEst'])
blenEst.load_state_dict(model_dict['state_dict_blenEst'])

transform = transforms.Compose([transforms.ToTensor()])
revtransf = transforms.Compose([transforms.ToPILImage()])

L1_lossFn = models.L1Loss()

img0 = io.imread(args.img0Path).astype(np.float32)/255.0
img1 = io.imread(args.img1Path).astype(np.float32)/255.0
sket = io.imread(args.sketPath)[np.newaxis, :, :].astype(np.float32)/255.0

img0 = transform(img0).unsqueeze(0).to(device)
img1 = transform(img1).unsqueeze(0).to(device)
sket = torch.from_numpy(sket).unsqueeze(0).to(device)

# flow estimation
imgt_temp = netT(torch.cat((sket, img0, img1), dim = 1))
featSkt = sketExt(imgt_temp)
featIg0 = imagExt(img0)
featIg1 = imagExt(img1)

f_t0 = flowEst(featSkt, featIg0)
f_t1 = flowEst(featSkt, featIg1)

f_0t = flowEst(featIg0, featSkt)
f_1t = flowEst(featIg1, featSkt)

I_t0 = flowBackWarp(img0, f_t0)
I_t1 = flowBackWarp(img1, f_t1)

O_t0 = occlusiCheck(f_t0, f_0t)
O_t1 = occlusiCheck(f_t1, f_1t)

W_0 = blenEst(torch.cat((I_t0, I_t1, O_t0, O_t1, sket), dim = 1))

W_1 = 1 - W_0
I_t = W_0 * I_t0 + W_1 * I_t1
  
io.imsave(args.saveImgPath, np.asarray(revtransf(I_t.cpu()[0])))
