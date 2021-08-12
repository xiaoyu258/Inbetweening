import torch
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torch.nn.functional as F
import torchvision.transforms as transforms

import skimage.io as io
import numpy as np
from skimage import morphology
from scipy import ndimage
import math

transform = transforms.Compose([transforms.ToTensor()])
revtransf = transforms.Compose([transforms.ToPILImage()])

class Network(torch.nn.Module):
    def __init__(self, gpu=None):
        super(Network, self).__init__()

        self.moduleVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.moduleCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )
        
        self.gpu = gpu
        if gpu is not None:
            self.cuda(gpu)
    # end

    def forward(self, tensorInput):
        
        if self.gpu is not None:
            tensorInput = tensorInput.cuda(self.gpu)
            
        tensorBlue = (tensorInput[:, 0:1, :, :] * 255.0) - 127.5
        tensorGreen = (tensorInput[:, 1:2, :, :] * 255.0) - 127.5
        tensorRed = (tensorInput[:, 2:3, :, :] * 255.0) - 127.5

        tensorInput = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 1)

        vggOne = self.moduleVggOne(tensorInput)
        vggTwo = self.moduleVggTwo(vggOne)
        vggThr = self.moduleVggThr(vggTwo)
        vggFou = self.moduleVggFou(vggThr)
        vggFiv = self.moduleVggFiv(vggFou)

        scoreOne = self.moduleScoreOne(vggOne)
        scoreTwo = self.moduleScoreTwo(vggTwo)
        scoreThr = self.moduleScoreThr(vggThr)
        scoreFou = self.moduleScoreFou(vggFou)
        scoreFiv = self.moduleScoreFiv(vggFiv)
        
        H = tensorInput.size(2)
        W = tensorInput.size(3)

        scoreOne = torch.nn.functional.interpolate(input=scoreOne, size=(H, W), mode='bilinear', align_corners=False)
        scoreTwo = torch.nn.functional.interpolate(input=scoreTwo, size=(H, W), mode='bilinear', align_corners=False)
        scoreThr = torch.nn.functional.interpolate(input=scoreThr, size=(H, W), mode='bilinear', align_corners=False)
        scoreFou = torch.nn.functional.interpolate(input=scoreFou, size=(H, W), mode='bilinear', align_corners=False)
        scoreFiv = torch.nn.functional.interpolate(input=scoreFiv, size=(H, W), mode='bilinear', align_corners=False)
        scoreFin = self.moduleCombine(torch.cat([ scoreOne, scoreTwo, scoreThr, scoreFou, scoreFiv ], 1))

        return F.sigmoid(1 - scoreTwo)

##########################################################

torch.set_grad_enabled(False)       # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EdgeDet = Network(gpu = 0)
EdgeDet.load_state_dict(torch.load('./checkpoints/hed.pkl'))

framDir = './dataset/frame'
saveDir = './dataset/dismap'

for clipDir in sorted(os.listdir(framDir)):
    
    clipPath = os.path.join(framDir, clipDir)
    savePath = os.path.join(saveDir, clipDir)
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    
    for index, frame in enumerate(sorted(os.listdir(clipPath))):
        
        inDir = '%s%s%s%s%s' % (framDir, '/', clipDir, '/', frame)
        ouDir = '%s%s%s%s%s%s' % (saveDir, '/', clipDir, '/', frame.split('.')[0], '.npy')

        imgt = io.imread(inDir).astype(np.float32)/255.0
        imgt = transform(imgt)
        imgt = imgt.unsqueeze(0).to(device)

        edgmap = np.asarray(revtransf(EdgeDet(imgt).cpu()[0]))
        edgmap = edgmap / 255.0

        newmap = np.zeros(edgmap.shape)
        newmap[edgmap > 1.0/(1 + math.exp(-0.5))] = 1.0            
        dismap = ndimage.distance_transform_edt(newmap)

        np.save(ouDir, dismap)
