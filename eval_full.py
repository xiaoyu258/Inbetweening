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
import model_deform


parser = argparse.ArgumentParser()
parser.add_argument("--img0Path", type=str, default='./images/f_032242.png', help='input image0 path')
parser.add_argument("--img1Path", type=str, default='./images/f_032246.png', help='input image1 path')
parser.add_argument("--sketPath", type=str, default='./images/f_032244_sket.png', help='input sketch_t path')
parser.add_argument("--saveDir", type=str, default='./images', help='saved image It path')
parser.add_argument("--modelPath", type=str, default='./checkpoints/full_model.ckpt', help='checkpoint path')
parser.add_argument("--frm_num", type=int, default=5, help='number of frame to be interpolated')
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
flowRef = models.UNet(14, 8)
ImagRef = model_deform.DeformUNet(21, 15)

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
    flowRef = nn.DataParallel(flowRef, device_ids=multiGPUs)
    ImagRef = nn.DataParallel(ImagRef, device_ids=multiGPUs)
    
    flowBackWarp = nn.DataParallel(flowBackWarp, device_ids=multiGPUs)
    occlusiCheck = nn.DataParallel(occlusiCheck, device_ids=multiGPUs)
    
netT = netT.to(device)
sketExt = sketExt.to(device)
imagExt = imagExt.to(device)
flowEst = flowEst.to(device)
blenEst = blenEst.to(device)
flowRef = flowRef.to(device)
ImagRef = ImagRef.to(device)

flowBackWarp = flowBackWarp.to(device)
occlusiCheck = occlusiCheck.to(device)

model_dict = torch.load(args.modelPath, map_location='cuda:0')
for key, value in model_dict.items():
    print(key)
    
netT.load_state_dict(model_dict['state_dict_netT'])
sketExt.load_state_dict(model_dict['state_dict_sketExt'])
imagExt.load_state_dict(model_dict['state_dict_imagExt'])
flowEst.load_state_dict(model_dict['state_dict_flowEst'])
blenEst.load_state_dict(model_dict['state_dict_blenEst'])
flowRef.load_state_dict(model_dict['state_dict_flowRef'])
ImagRef.load_state_dict(model_dict['state_dict_ImagRef'])

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

imgall = []
frm_list = [x / (args.frm_num + 1) for x in range(args.frm_num + 2)]

for frm_index, i in enumerate(frm_list):
    if i == 0:
        imgall.append(img0)
    if i == 1:
        imgall.append(img1)
    if i == 0.5:
        imgall.append(I_t)

    if i != 0 and i < 0.5:
        t = i * 2
        co_eff = [-t * (1 - t), t * t, (1 - t) * (1 - t), -t * (1 - t)]
        f_lr = f_0t
        f_rl = f_t0

        f_k0 = co_eff[0] * f_lr + co_eff[1] * f_rl
        f_kt = co_eff[2] * f_lr + co_eff[3] * f_rl
        f_0k = t * f_0t
        f_tk = (1 - t) * f_t0

        I_k0 = flowBackWarp(img0, f_k0)
        I_kt = flowBackWarp(I_t, f_kt)

        refflows = flowRef(torch.cat((I_k0, I_kt, f_k0, f_0k, f_kt, f_tk), dim = 1))

        f_k0 = refflows[:, 0:2, :, :] + f_k0
        f_0k = refflows[:, 2:4, :, :] + f_0k
        f_kt = refflows[:, 4:6, :, :] + f_kt
        f_tk = refflows[:, 6:8, :, :] + f_tk

        I_k0 = flowBackWarp(img0, f_k0)
        I_kt = flowBackWarp(I_t, f_kt)

        O_k0 = occlusiCheck(f_k0, f_0k)
        O_kt = occlusiCheck(f_kt, f_tk)

        W_0 = blenEst(torch.cat((I_k0, I_kt, O_k0, O_kt, flowBackWarp(sket, f_kt)), dim = 1))
        W_1 = 1 - W_0
        I_k = W_0 * I_k0 + W_1 * I_kt

        imgall.append(I_k)

    if i != 1 and i > 0.5:
        t = (i - 0.5) * 2
        co_eff = [-t * (1 - t), t * t, (1 - t) * (1 - t), -t * (1 - t)]
        f_lr = f_t1
        f_rl = f_1t

        f_kt = co_eff[0] * f_lr + co_eff[1] * f_rl
        f_k1 = co_eff[2] * f_lr + co_eff[3] * f_rl
        f_tk = t * f_t1
        f_1k = (1 - t) * f_1t

        I_kt = flowBackWarp(I_t, f_kt)
        I_k1 = flowBackWarp(img1, f_k1)

        refflows = flowRef(torch.cat((I_k1, I_kt, f_k1, f_1k, f_kt, f_tk), dim = 1))

        f_k1 = refflows[:, 0:2, :, :] + f_k1
        f_1k = refflows[:, 2:4, :, :] + f_1k
        f_kt = refflows[:, 4:6, :, :] + f_kt
        f_tk = refflows[:, 6:8, :, :] + f_tk

        I_kt = flowBackWarp(I_t, f_kt)
        I_k1 = flowBackWarp(img1, f_k1)
        O_kt = occlusiCheck(f_kt, f_tk)
        O_k1 = occlusiCheck(f_k1, f_1k)

        W_0 = blenEst(torch.cat((I_kt, I_k1, O_kt, O_k1, flowBackWarp(sket, f_kt)), dim = 1))
        W_1 = 1 - W_0
        I_k = W_0 * I_kt + W_1 * I_k1

        imgall.append(I_k)

imgref = torch.clamp(ImagRef(torch.cat(imgall, dim = 1)) + torch.cat(imgall[1:-1], dim = 1), 0, 1)

for i in range(len(imgall) - 2):
    io.imsave('%s%s%s%s' % (args.saveDir, '/f_output_', str(i + 1), '.png'), np.asarray(revtransf(imgref[:,(i*3):(i+1)*3,:,:].cpu()[0])))

    