import torch.utils.data as data
import os
import skimage.io as io
import numpy as np
import torch
import config
    
class SuperSloMo(data.Dataset):
    def __init__(self, root, transform=None, train=True):
        
        # Populate the list with image paths for all the frame in `root`.
        
        indexSum = 0
        framesPath = []
        sketchPath = []
        dismapPath = []
        
        if train:
            clips = os.listdir(os.path.join(root, 'frame'))[0:-100]
        else:
            clips = os.listdir(os.path.join(root, 'frame'))[-100:]
        
        for folder in clips:
            
            clipsPath = os.path.join(root, 'frame', folder)
            sktchPath = os.path.join(root, 'sketch', folder)
            dimapDir = os.path.join(root, 'dismap', folder)
            
            frameList = sorted(os.listdir(clipsPath))
            
            indexNum  = int(len(frameList) / 3)
            
            for i in range(indexNum):
                framesPath.append([])
                sketchPath.append([])
                dismapPath.append([])
                
                for j in range(3):
                    framesPath[indexSum].append(os.path.join(clipsPath, frameList[i * 3 + j]))

                sketchPath[indexSum].append(os.path.join(sktchPath, frameList[i * 3 + 1]))
                dismapPath[indexSum].append(os.path.join(dimapDir, frameList[i * 3 + 1].split('.')[0] + '.npy'))

                indexSum = indexSum + 1

        self.root = root
        self.transform = transform
        self.framesPath = framesPath
        self.sketchPath = sketchPath
        self.dismapPath = dismapPath

    def __getitem__(self, index):
        
        sktt = torch.from_numpy(io.imread(self.sketchPath[index][0])[np.newaxis, :, :].astype(np.float32)/255.0)
                
        img0 = io.imread(self.framesPath[index][0]).astype(np.float32)/255.0
        imgt = io.imread(self.framesPath[index][1]).astype(np.float32)/255.0
        img1 = io.imread(self.framesPath[index][2]).astype(np.float32)/255.0
        
        dismap = torch.from_numpy(np.load(self.dismapPath[index][0])[np.newaxis, :, :].astype(np.float32))
        
        if self.transform is not None:
            img0 = self.transform(img0)
            imgt = self.transform(imgt)
            img1 = self.transform(img1)
        
        return img0, imgt, img1, sktt, dismap


    def __len__(self):
        return len(self.framesPath)
    
