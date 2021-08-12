import torch.utils.data as data
import os
import skimage.io as io
import numpy as np
import torch
    
class SuperSloMo(data.Dataset):
    def __init__(self, root, transform=None, train=True, frm_num = 0):
        
        # Populate the list with image paths for all the frame in `root`.
        
        indexSum = 0
        framesPath = []
        sketchPath = []
        
        if train:
            clips = os.listdir(os.path.join(root, 'frame'))[0:-100]
        else:
            clips = os.listdir(os.path.join(root, 'frame'))[-100:]
        
        for folder in clips:
            
            clipsPath = os.path.join(root, 'frame', folder)
            sktchPath = os.path.join(root, 'sketch', folder)
            
            frameList = sorted(os.listdir(clipsPath))
            sketcList = sorted(os.listdir(sktchPath))
            
            indexNum  = int((len(frameList) - 1) / (frm_num - 1))
            
            for i in range(indexNum):
                framesPath.append([])
                sketchPath.append([])
                
                for j in range(frm_num):
                    framesPath[indexSum].append(os.path.join(clipsPath, frameList[i * (frm_num - 1) + j]))

                sketchPath[indexSum].append(os.path.join(sktchPath, frameList[int(i * (frm_num - 1) + (frm_num - 1) / 2)]))

                indexSum = indexSum + 1

        self.root = root
        self.transform = transform
        self.framesPath = framesPath
        self.sketchPath = sketchPath

    def __getitem__(self, index):
        imgs = []
        for i in range(len(self.framesPath[index])):
            img = io.imread(self.framesPath[index][i]).astype(np.float32)/255.0
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
            
        if np.random.randint(2):
            imgs.reverse()

        sktt = torch.from_numpy(io.imread(self.sketchPath[index][0])[np.newaxis, :, :].astype(np.float32)/255.0)
        
        return imgs, sktt


    def __len__(self):
        return len(self.framesPath)
    
