import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import models
import model_hed
import dataloader_syn

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default='./dataset')
parser.add_argument("--checkpt_dir", type=str, default='./checkpoints')
parser.add_argument("--cont", type=bool, default=False, help='set to True and set `checkpoint` path.')
parser.add_argument("--contpt", type=str, default='./checkpoints/.ckpt', help='path of checkpoint for pretrained model')
parser.add_argument("--init_lr", type=float, default=0.0001, help='set initial learning rate.')
parser.add_argument("--epochs", type=int, default=100, help='number of epochs to train.')
parser.add_argument("--batch_size", type=int, default=4, help='batch size for training.')
parser.add_argument("--checkpoint_epoch", type=int, default=1, help='checkpoint saving frequency. N: after every N epochs.')
args = parser.parse_args()

log_name = './log/frame_synthesis'
cpt_name = '/frame_synthesis_model_'

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

W = 576
H = 384
flowBackWarp = models.backWarp(W, H)
occlusiCheck = models.occlusionCheck(W, H)
EdgeDet = model_hed.Network()
EdgeDet.load_state_dict(torch.load('./checkpoints/hed.pkl'))

if args.cont == False:
    pretrained_dict = torch.load('./checkpoints/pwcnet.pkl')
    sketExt.load_state_dict(pretrained_dict, strict=False)
    imagExt.load_state_dict(pretrained_dict, strict=False)
    flowEst.load_state_dict(pretrained_dict, strict=False)

if torch.cuda.device_count() >= 1:
    netT = nn.DataParallel(netT, device_ids=multiGPUs)
    sketExt = nn.DataParallel(sketExt, device_ids=multiGPUs)
    imagExt = nn.DataParallel(imagExt, device_ids=multiGPUs)
    flowEst = nn.DataParallel(flowEst, device_ids=multiGPUs)
    blenEst = nn.DataParallel(blenEst, device_ids=multiGPUs)
    
    flowBackWarp = nn.DataParallel(flowBackWarp, device_ids=multiGPUs)
    occlusiCheck = nn.DataParallel(occlusiCheck, device_ids=multiGPUs)
    EdgeDet = nn.DataParallel(EdgeDet, device_ids=multiGPUs)
    
netT = netT.to(device)
sketExt = sketExt.to(device)
imagExt = imagExt.to(device)
flowEst = flowEst.to(device)
blenEst = blenEst.to(device)

flowBackWarp = flowBackWarp.to(device)
occlusiCheck = occlusiCheck.to(device)
EdgeDet = EdgeDet.to(device)

for param in EdgeDet.parameters():
    param.requires_grad = False
    
# Load Datasets 
transform = transforms.Compose([transforms.ToTensor()])
revtransf = transforms.Compose([transforms.ToPILImage()])

trainset = dataloader_syn.SuperSloMo(root=args.dataset_dir, transform=transform, train=True)
train_sample = torch.utils.data.sampler.RandomSampler(trainset)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler = train_sample, drop_last = True)

validationset = dataloader_syn.SuperSloMo(root=args.dataset_dir, transform=transform, train=False)
valid_loader = torch.utils.data.DataLoader(validationset, batch_size=args.batch_size, drop_last = True)

print('train_loader: ',len(train_loader), '\t', args.batch_size, '\t', len(train_loader)*args.batch_size)
print('valid_loader: ',len(valid_loader), '\t', args.batch_size, '\t', len(valid_loader)*args.batch_size)

# Loss and Optimizer
L1_lossFn = models.L1Loss()

params = list(netT.parameters()) + list(sketExt.parameters()) + list(imagExt.parameters()) + list(flowEst.parameters()) + list(blenEst.parameters())
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
else:
    start_epoch = 0
    
    
# Validation function 
def validate():
    
    tloss = 0
    
    with torch.no_grad():
        
        for validationIndex, (img0, imgt, img1, sktt, dismap) in enumerate(valid_loader):

            img0 = img0.to(device)
            imgt = imgt.to(device)
            img1 = img1.to(device)
            sktt = sktt.to(device)

            # Calculate flow between reference frames I0 and I1
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

            tloss += L1_lossFn(I_t, imgt).item()
            
    return (tloss / len(valid_loader))


# Main training loop
for epoch in range(start_epoch, args.epochs):
    
    sumWarpsLoss = 0
    sumReconLoss = 0
    sumEdgesLoss = 0
    sumTrainLoss = 0

    for trainIndex, (img0, imgt, img1, sktt, dismap) in enumerate(train_loader):
        
        t0 = time.time()
        
        img0 = img0.to(device)
        imgt = imgt.to(device)
        img1 = img1.to(device)
        sktt = sktt.to(device)
        dismap = dismap.to(device)
        
        optimizer.zero_grad()
        
        # Calculate flow between reference frames I0 and I1
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
       
        # Recn Loss
        recnLoss = L1_lossFn(I_t, imgt)
        
        # Warp Loss
        warpLoss = (L1_lossFn(flowBackWarp(img0, f_t0), imgt) + L1_lossFn(flowBackWarp(img1, f_t1), imgt) + \
                L1_lossFn(flowBackWarp(imgt, f_0t), img0) + L1_lossFn(flowBackWarp(imgt, f_1t), img1)) * 0.5
        
        # Edge Loss
        edge = EdgeDet(I_t)
        edgeLoss = torch.mean(dismap * (1 - edge)) * 0.01
        
        loss = recnLoss + warpLoss + edgeLoss
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        sumReconLoss += recnLoss.item()
        sumWarpsLoss += warpLoss.item()
        sumEdgesLoss += edgeLoss.item()
        sumTrainLoss += loss.item()

        print("Epoch: ", epoch, "Iteration: ", trainIndex, 'recnLoss: ', recnLoss.item(), 'warpLoss: ', warpLoss.item(), 'edgeLoss: ', edgeLoss.item()) 
              
        t1 = time.time()
        print("time: ", t1-t0)
        
    #Tensorboard
    vLoss = validate()
    itr = (epoch + 1) * len(train_loader)
    writer.add_scalars('Loss', {'trainLoss': sumTrainLoss/len(train_loader),
                        'warpLoss': sumWarpsLoss/len(train_loader),
                        'recnLoss': sumReconLoss/len(train_loader),
                        'edgeLoss': sumEdgesLoss/len(train_loader),
                        'validLoss': vLoss}, itr)

    sumWarpsLoss = 0
    sumReconLoss = 0
    sumEdgesLoss = 0
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
                'state_dict_blenEst': blenEst.state_dict()}
        torch.save(dict1, args.checkpt_dir + cpt_name + str(epoch) + ".ckpt")
