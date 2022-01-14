

import torch
import torch.nn as neuralNet
import  torchvision.transforms.functional as tf

class TwoConvolutionalLayers(neuralNet.Module):
    def __init__(self, in_channels,out_channels):
        super(TwoConvolutionalLayers, self).__init__()
        self.conv = neuralNet.Sequential(
            neuralNet.Conv2d(in_channels,out_channels,3,1,1,bias=False),
            neuralNet.BatchNorm2d(out_channels),
            neuralNet.ReLU(inplace=True),
            neuralNet.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            neuralNet.BatchNorm2d(out_channels),
            neuralNet.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)

class UNET(neuralNet.Module):
    def __init__(self,in_channels = 3, out_channels = 1, features = [64,128, 256, 512]):
        super(UNET,self).__init__()
        self.downsampling = neuralNet.ModuleList()
        self.upsampling = neuralNet.ModuleList()
        self.pool = neuralNet.MaxPool2d(kernel_size=2, stride=2)

        for f in features:
            self.downsampling.append(TwoConvolutionalLayers(in_channels,f))
            in_channels = f

        for f in reversed(features):
            self.upsampling.append(neuralNet.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.upsampling.append(TwoConvolutionalLayers(f*2,f))

        self.bn = TwoConvolutionalLayers(features[-1],features[-1]*2)
        self.final = neuralNet.Conv2d(features[0],out_channels, kernel_size=1)

    def forward(self,x):
        skips = []
        for ds in self.downsampling:
            x = ds(x)
            skips.append(x)
            x =  self.pool(x)


        x = self.bn(x)
        skips = skips[::-1]
        for i in range(0, len(self.upsampling),2):
            x = self.upsampling[i](x)

            skip = skips[i//2]

            if x.shape != skip.shape:
                x = tf.resize(x,size=skip.shape[2:])
            #print(skip.shape)

            #print(skip.shape)
            ns = (skip, x)
            concatenate_skip = torch.cat(ns, dim=1)
            x = self.upsampling[i+1](concatenate_skip)

        return  self.final(x)

def test():
    x= torch.randn((3,1,160,160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    #print(preds.shape)
    #print(x.shape)
    assert  preds.shape == x.shape

if __name__ == "__main__":
    test()