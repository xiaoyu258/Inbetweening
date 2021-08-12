import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import model_deform
import models
import dataloader_full

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default='./dataset')
parser.add_argument("--checkpt_dir", type=str, default='./checkpoints')
parser.add_argument("--cont", type=bool, default=False, help='set to True and set `checkpoint` path.')
parser.add_argument("--contpt", type=str, default='./checkpoints/.ckpt', help='path of checkpoint for pretrained model')
parser.add_argument("--init_lr", type=float, default=0.0001, help='set initial learning rate.')
parser.add_argument("--epochs", type=int, default=100, help='number of epochs to train.')
parser.add_argument("--batch_size", type=int, default=4, help='batch size for training.')
parser.add_argument("--checkpoint_epoch", type=int, default=5, help='checkpoint saving frequency. N: after every N epochs.')
parser.add_argument("--frm_num", type=int, default=7)
args = parser.parse_args()

log_name = './log/full_model'
cpt_name = '/full_model_' 

writer = SummaryWriter(log_name)

print("torch.cuda.is_available: ", torch.cuda.is_available())
print("torch.cuda.device_count: ", torch.cuda.device_count())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
multiGPUs = [0, 1, 2, 3]

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
    
# Load Datasets 
transform = transforms.Compose([transforms.ToTensor()])
revtransf = transforms.Compose([transforms.ToPILImage()])

trainset = dataloader_full.SuperSloMo(root=args.dataset_dir, transform=transform, train=True, frm_num = args.frm_num)
train_sample = torch.utils.data.sampler.RandomSampler(trainset)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler = train_sample, drop_last = True)

validationset = dataloader_full.SuperSloMo(root=args.dataset_dir, transform=transform, train=False, frm_num = args.frm_num)
valid_loader = torch.utils.data.DataLoader(validationset, batch_size=args.batch_size, drop_last = True)

print('train_loader: ',len(train_loader), '\t', args.batch_size, '\t', len(train_loader)*args.batch_size)
print('valid_loader: ',len(valid_loader), '\t', args.batch_size, '\t', len(valid_loader)*args.batch_size)

# Loss and Optimizer
L1_lossFn = models.L1Loss()

params = list(netT.parameters()) + list(sketExt.parameters()) + list(imagExt.parameters()) + list(flowEst.parameters()) + list(blenEst.parameters()) + list(flowRef.parameters()) + list(ImagRef.parameters())

optimizer = optim.Adam(params, lr=args.init_lr)

if args.cont:
    model_dict = torch.load(args.contpt)
    start_epoch = model_dict['epoch'] + 1
    optimizer.load_state_dict(model_dict['optim_state_dict'])
    netT.load_state_dict(model_dict['state_dict_netT'])
    sketExt.load_state_dict(model_dict['state_dict_sketExt'])
    imagExt.load_state_dict(model_dict['state_dict_imagExt'])
    flowEst.load_state_dict(model_dict['state_dict_flowEst'])
    blenEst.load_state_dict(model_dict['state_dict_blenEst'])
    flowRef.load_state_dict(model_dict['state_dict_flowRef'])
    ImagRef.load_state_dict(model_dict['state_dict_ImagRef'])
else:
    start_epoch = 0
    model_dict = torch.load('./checkpoints/frame_synthesis_model.ckpt')
    netT.load_state_dict(model_dict['state_dict_netT'])
    sketExt.load_state_dict(model_dict['state_dict_sketExt'])
    imagExt.load_state_dict(model_dict['state_dict_imagExt'])
    flowEst.load_state_dict(model_dict['state_dict_flowEst'])
    blenEst.load_state_dict(model_dict['state_dict_blenEst'])
    
# Validation function 
def validate():
    
    tloss = 0
    
    with torch.no_grad():
        
        for validationIndex, (imgs, sktt) in enumerate(valid_loader):

            for i in range(len(imgs)):
                imgs[i] = imgs[i].to(device)

            sktt = sktt.to(device)

            img0 = imgs[0]
            img1 = imgs[-1]

            imgt_temp = netT(torch.cat((sktt, img0, img1), dim = 1))
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

            W_0 = blenEst(torch.cat((I_t0, I_t1, O_t0, O_t1, sktt), dim = 1))

            W_1 = 1 - W_0
            I_t = W_0 * I_t0 + W_1 * I_t1

            imgall = []
            frm_list = [x / (args.frm_num - 1) for x in range(args.frm_num)]

            for frm_index, i in enumerate(frm_list):
                if i == 0:
                    imgall.append(imgs[frm_index])
                if i == 1:
                    imgall.append(imgs[frm_index])
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

                    W_0 = blenEst(torch.cat((I_k0, I_kt, O_k0, O_kt, flowBackWarp(sktt, f_kt)), dim = 1))
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

                    W_0 = blenEst(torch.cat((I_kt, I_k1, O_kt, O_k1, flowBackWarp(sktt, f_kt)), dim = 1))
                    W_1 = 1 - W_0
                    I_k = W_0 * I_kt + W_1 * I_k1

                    imgall.append(I_k)

            imgres = imgall[1:-1]

            imgref = ImagRef(torch.cat(imgall, dim = 1)) + torch.cat(imgres, dim = 1)

            recnLoss = L1_lossFn(imgref, torch.cat(imgs[1:-1], dim = 1))

            tloss += recnLoss.item()  
            
    return tloss / len(valid_loader)

# Main training loop

for epoch in range(start_epoch, args.epochs):
    
    sumReconLoss = 0
    sumTrainLoss = 0

    for trainIndex, (imgs, sktt) in enumerate(train_loader):
        
        t0 = time.time()
        
        for i in range(len(imgs)):
            imgs[i] = imgs[i].to(device)
        
        sktt = sktt.to(device)
        
        optimizer.zero_grad()
        
        # Calculate flow between reference frames I0 and I1
        img0 = imgs[0]
        img1 = imgs[-1]
        
        imgt_temp = netT(torch.cat((sktt, img0, img1), dim = 1))
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

        W_0 = blenEst(torch.cat((I_t0, I_t1, O_t0, O_t1, sktt), dim = 1))
        
        W_1 = 1 - W_0
        I_t = W_0 * I_t0 + W_1 * I_t1
        
        imgall = []
        frm_list = [x / (args.frm_num - 1) for x in range(args.frm_num)]
        
        for frm_index, i in enumerate(frm_list):
            if i == 0:
                imgall.append(imgs[frm_index])
            if i == 1:
                imgall.append(imgs[frm_index])
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

                W_0 = blenEst(torch.cat((I_k0, I_kt, O_k0, O_kt, flowBackWarp(sktt, f_kt)), dim = 1))
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

                W_0 = blenEst(torch.cat((I_kt, I_k1, O_kt, O_k1, flowBackWarp(sktt, f_kt)), dim = 1))
                W_1 = 1 - W_0
                I_k = W_0 * I_kt + W_1 * I_k1
                
                imgall.append(I_k)
        
        imgres = imgall[1:-1]
        
        imgref = ImagRef(torch.cat(imgall, dim = 1)) + torch.cat(imgres, dim = 1)
        
        recnLoss = L1_lossFn(imgref, torch.cat(imgs[1:-1], dim = 1))

        loss = recnLoss
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        sumReconLoss += recnLoss.item()
        sumTrainLoss += loss.item()

        print("Epoch: ", epoch, "Iteration: ", trainIndex, 'recnLoss: ', recnLoss.item()) 
              
        t1 = time.time()
        print("time: ", t1-t0)
        
    #Tensorboard
    
    vLoss = validate()
    
    itr = (epoch + 1) * len(train_loader)

    writer.add_scalars('Loss', {'trainLoss': sumTrainLoss/len(train_loader),
                        'recnLoss': sumReconLoss/len(train_loader),
                        'validationLoss': vLoss}, itr)

    sumReconLoss = 0
    sumTrainLoss = 0
    
    # Create checkpoint after every `args.checkpoint_epoch` epochs
    if ((epoch % args.checkpoint_epoch) == args.checkpoint_epoch - 1):
        dict1 = {
                'epoch':epoch,
                'BatchSz':args.batch_size,
                'optim_state_dict': optimizer.state_dict(),
                'state_dict_netT': netT.state_dict(),
                'state_dict_sketExt': sketExt.state_dict(),
                'state_dict_imagExt': imagExt.state_dict(),
                'state_dict_flowEst': flowEst.state_dict(),
                'state_dict_blenEst': blenEst.state_dict(),
                'state_dict_flowRef': flowRef.state_dict(),
                'state_dict_ImagRef': ImagRef.state_dict()}
        torch.save(dict1, args.checkpt_dir + cpt_name + str(epoch) + ".ckpt")
