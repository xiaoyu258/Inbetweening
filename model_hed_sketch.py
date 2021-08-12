import torch
import torch.nn.functional as F

arguments_strModel = './checkpoints/hed.pkl'

class Network(torch.nn.Module):
    def __init__(self):
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
            #torch.nn.Sigmoid()
        )

        self.load_state_dict(torch.load(arguments_strModel))
    # end

    def forward(self, tensorInput):
        tensorBlue = (tensorInput[:, 0:1, :, :]) - 127.5
        tensorGreen = (tensorInput[:, 1:2, :, :]) - 127.5
        tensorRed = (tensorInput[:, 2:3, :, :]) - 127.5

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

        return scoreTwo