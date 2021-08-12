import os
import PIL
import torch
import numpy as np
import skimage.io as io
import skimage.transform as tf
from skimage import morphology
from skimage.util import img_as_ubyte
from scipy import ndimage

import model_hed_sketch

torch.set_grad_enabled(False)       
sktNetwork = model_hed_sketch.Network().cuda().eval()
    
def genSkt(inpDir, savDir):
    
    inpName = os.path.basename(os.path.abspath(inpDir)).split('.')[0]

    sktDir = savDir + '/' + inpName + '.bmp'
    svgDir = savDir + '/' + inpName + '.svg'
    pngDir = savDir + '/' + inpName + '_temp.png'
    finDir = savDir + '/' + inpName + '.png'

    inImg = io.imread(inpDir)[:, :, ::-1]
    
    scaleFat = np.sqrt((1024 * 1792) / (inImg.shape[0] * inImg.shape[1]))
    inImg = img_as_ubyte(tf.rescale(inImg, scaleFat))
    inImg = torch.FloatTensor(inImg.transpose(2, 0, 1).astype(np.float32))

    H = inImg.size(1)
    W = inImg.size(2)

    inImg = inImg.cuda().view(1, 3, H, W)
    ouImg = sktNetwork(inImg)[0].cpu()
    ouImg = 1.0 - ouImg.clamp(0.0, 1.0).numpy()[0]
    ouImg[ouImg<1] = 0

    PIL.Image.fromarray((ouImg*255.0).astype(np.uint8)).save(sktDir)

    exeSvg = 'F:/Downloads/potrace/potrace.exe'
    cmdSvg = '%s%s%s%s%s%s' % (exeSvg, ' "', sktDir, '" -o "', svgDir, '" --svg \n')
    os.system(cmdSvg)

    exePng = 'magick'
    cmdPng = '%s%s%s%s%s%s' % (exePng, ' -density 72 "', svgDir, '" "', pngDir, '" \n')
    os.system(cmdPng)

    img = io.imread(pngDir)

    holeSize = 20 / (672 * 384) * H * W

    inpImg = np.zeros(img.shape, dtype = np.uint8)
    outImg = np.zeros(img.shape, dtype = np.uint8)

    inpImg[img == 255] = 255
    rmImg = morphology.remove_small_holes(inpImg, holeSize)
    outImg[rmImg == True] = 255
    img[(outImg - inpImg) == 255] = 255
    img = img_as_ubyte(tf.rescale(img, 1.0/scaleFat))

    io.imsave(finDir, img)
    os.remove(sktDir)
    os.remove(svgDir)
    os.remove(pngDir)
    