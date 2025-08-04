import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import sys

from IPython.core import debugger
debug = debugger.Pdb().set_trace

class Global(nn.Module):
    def __init__(self):
        super(Global, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size = [1,7], stride = [1,4], padding = [0,3]),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(64, 64, kernel_size = [1,3], stride = [1,2], padding = [0,1]),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace = True),
                                   )
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = [1,3], stride = [1,2], padding = [0,1]),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(64, 64, kernel_size = [1,3], stride = [1,2], padding = [0,1]),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace = True),
                                    #nn.MaxPool2d(kernel_size = [2,2], stride = 2),
                                   )
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size = [3,3], stride = 1, padding = [1,1]),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(128, 128, kernel_size = [3,3], stride = 2, padding = [1,1]),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace = True),
                                   ) 
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = [3,3], stride = 1, padding = [1,1]),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(256, 256, kernel_size = [3,3], stride = 2, padding = [1,1]),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace = True),
                                   ) 
        self.layer5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size = [3,3], stride = 1, padding = [1,1]),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(512, 512, kernel_size = [3,3], stride = 2, padding = [1,1]),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace = True),
                                   )
        self.layer6 = nn.Sequential(nn.Conv2d(512, 128, kernel_size = [4,4]),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace = True),
                                   )                                                                 
    def forward(self, x):
        b,c,h,w = x.shape
        x = x.view(b*c,-1,h,w)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        #out = out.view(b,-1)             
        return out


class Neighbor(nn.Module):
    def __init__(self, can):
        super(Neighbor, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2)
        self.norm1 = nn.InstanceNorm3d(can)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=5, padding=2)
        self.norm2 = nn.InstanceNorm3d(can)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=5, padding=2)
        self.norm3 = nn.InstanceNorm3d(can)
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x):
        b,c,h,w = x.shape
        x = x.view(b*c,-1,h,w)
        
        y = self.conv1(x).view(b,c,-1,h,w)
        y = self.relu(self.norm1(y)).view(b*c,-1,h,w)

        y = self.conv2(y).view(b,c,-1,h,w)
        y = self.relu(self.norm2(y)).view(b*c,-1,h,w)
        
        y = self.conv3(y).view(b,c,-1,h,w)
        y = self.relu(self.norm3(y)).view(b*c,-1,h,w)
        return y


class Encoder(nn.Module):    
    def __init__(self, can, obs):
        super(Encoder, self).__init__()
        self.can = can
        self.obs = obs

        self.neig = Neighbor(can)
        self.glob = Global()

        self.relu = nn.ReLU(inplace = True)
        c_ind = torch.eye(can,can).repeat(1,obs).reshape(-1).view(-1,can)
        o_ind = torch.eye(obs,obs).repeat(can,1)
        self.ind = torch.cat((c_ind,o_ind),1).unsqueeze(0).view(1,-1,can+obs)
        
    def forward(self, x):
        b,c,h,w = x.shape
        neig_ = self.neig(x)
        glob_ = self.glob(x).squeeze()
        glob_ = glob_.view(b,self.can,1,-1).repeat(1,1,self.obs,1).view(b,self.can*self.obs,-1)
        neig_ = neig_.view(b,-1,w)
#         y = torch.cat((glob_,self.ind.repeat(b,1,1).cuda(),neig_),2)
        y = torch.cat((glob_,self.ind.repeat(b,1,1),neig_),2)
        return y
        

class Generator(nn.Module):
    def __init__(self, can=20, obs=32):
        super(Generator, self).__init__()
        self.num = can * obs
        self.lin1 = nn.Linear(1592, 1024)  #############focus on 8530 , it can be changed
        self.bn1 = nn.BatchNorm1d(self.num)
        self.lin2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(self.num)
        self.lin3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(self.num)
        self.lin4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(self.num)
        self.lin5 = nn.Linear(256, 256)
        self.bn5 = nn.BatchNorm1d(self.num)       
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        batch = x.size(0)
        y = self.relu(self.bn1(self.lin1(x)))
        y = self.relu(self.bn2(self.lin2(y)))
        y = self.relu(self.bn3(self.lin3(y)))
        y = self.relu(self.bn4(self.lin4(y)))
        y = self.relu(self.bn5(self.lin5(y)))
        y = y.view(batch, self.num, 16, 16)
        return y


class Decoder(nn.Module):
    def __init__(self, num):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(num, 128, kernel_size = [4,4], stride = 2, padding = [1,1]),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(128, 128, kernel_size = [3,3], stride = 1, padding = [1,1]),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace = True),
                                   ) 
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size = [4,4], stride = 2, padding = [1,1]),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(64, 64, kernel_size = [3,3], stride = 1, padding = [1,1]),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace = True),
                                   ) 
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size = [4,4], stride = 2, padding = [1,1]),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(32, 32, kernel_size = [3,3], stride = 1, padding = [1,1]),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace = True),
                                   )                    
        self.layer4_1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace = True),
                                     )
        self.layer4_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=3, dilation=3),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace = True),
                                     )
        self.layer4_3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=5, dilation=5),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace = True),
                                     )
        self.layer4_4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=7, dilation=7),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace = True),
                                     )                                                                                       
        self.layer5 = nn.Sequential(nn.Conv2d(128, 1, kernel_size = [1,1], stride = 1),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid(),
                                    #nn.ReLU(inplace = True),
                                   )
        self.interp = nn.Upsample(size=[128, 128], mode='bilinear', align_corners=True)

        self.dsn1 = nn.Conv2d(128, 1, 3)
        self.dsn2 = nn.Conv2d(64, 1, 3)
        self.dsn3 = nn.Conv2d(32, 1, 3)
        #self.sig = nn.Sigmoid()

    def forward(self, x):
        b,c,h,w = x.shape
        out = self.layer1(x)
        d1 = self.interp(self.dsn1(out))

        out = self.layer2(out)
        d2 = self.interp(self.dsn2(out))

        out = self.layer3(out)
        d3 = self.interp(self.dsn3(out))

        out1 = self.layer4_1(out)
        out2 = self.layer4_2(out)
        out3 = self.layer4_3(out)
        out4 = self.layer4_4(out)
        out = torch.cat((out1,out2,out3,out4),1)
        out = self.layer5(out)  #### already with sigmoid
        out = self.interp(out)

        return d1, d2, d3, out

### Below is from InversionNet
from math import ceil
import torch.nn.functional as F

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
        
class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# FlatFault/CurveFault
# Original InversionNet needs the input size to be exactly 5x1000x70 and output to be 70x70
# 5 is number of shots and 1000 is number of time samples and 70 is the number of grid size
class InversionNet5_1000_70to70_70(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
#         self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
#         self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, 9), padding=0).eval() ##added by YC, 2025/03/17

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        print('x0.shape',x.shape)
        x = self.convblock1(x) # (None, 32, 500, 70)
        print('x1.shape',x.shape)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        print('x2.shape',x.shape)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        print('x3.shape',x.shape)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        print('x4.shape',x.shape)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        print('x5.shape',x.shape)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        print('x7.shape',x.shape)
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        print('x8.shape',x.shape)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        print('x9.shape',x.shape)
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        print('x10.shape',x.shape)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        print('x11.shape',x.shape)
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        print('x12.shape',x.shape)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 
        print('x13.shape',x.shape)
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        print('x14.shape',x.shape)
        x = self.convblock8(x) # (None, 512, 1, 1)
        print('x15.shape',x.shape)
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        print('x16.shape',x.shape)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        print('x17.shape',x.shape)
        x = self.deconv2_1(x) # (None, 256, 10, 10) 
        print('x18.shape',x.shape)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        print('x19.shape',x.shape)
        x = self.deconv3_1(x) # (None, 128, 20, 20) 
        print('x20.shape',x.shape)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        print('x21.shape',x.shape)
        x = self.deconv4_1(x) # (None, 64, 40, 40) 
        print('x22.shape',x.shape)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        print('x23.shape',x.shape)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        print('x24.shape',x.shape)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        print('x25.shape',x.shape)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x

# The new adjusted InversionNet needs the input size to be exactly 32x50x1000 and output to be 100x100
# 30 is number of shots and 1000 is number of time samples and 100 is the number of grid size
class InversionNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(30, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
#         self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
#         self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
#         self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, 9), padding=0).eval() ##added by YC, 2025/03/17
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(1, 125), padding=0).eval() ##added by YC, 2025/03/17


        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=5, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=6, stride=3, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
#        print('x0.shape',x.shape)
        x = self.convblock1(x) # (None, 32, 500, 70)
#        print('x1.shape',x.shape)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
#        print('x2.shape',x.shape)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
#        print('x3.shape',x.shape)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
#        print('x4.shape',x.shape)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
#        print('x5.shape',x.shape)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
#        print('x7.shape',x.shape)
        x = self.convblock4_2(x) # (None, 128, 63, 70)
#        print('x8.shape',x.shape)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
#        print('x9.shape',x.shape)
        x = self.convblock5_2(x) # (None, 128, 32, 35)
#        print('x10.shape',x.shape)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
#        print('x11.shape',x.shape)
        x = self.convblock6_2(x) # (None, 256, 16, 18)
#        print('x12.shape',x.shape)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 
#        print('x13.shape',x.shape)
        x = self.convblock7_2(x) # (None, 256, 8, 9)
#        print('x14.shape',x.shape)
        x = self.convblock8(x) # (None, 512, 1, 1)
#        print('x15.shape',x.shape)
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
#        print('x16.shape',x.shape)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
#         print('x17.shape',x.shape)
        x = self.deconv2_1(x) # (None, 256, 10, 10) 
#         print('x18.shape',x.shape)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
#         print('x19.shape',x.shape)
        x = self.deconv3_1(x) # (None, 128, 20, 20) 
#         print('x20.shape',x.shape)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
#         print('x21.shape',x.shape)
        x = self.deconv4_1(x) # (None, 64, 41, 41) 
#         print('x22.shape',x.shape)
        x = self.deconv4_2(x) # (None, 64, 41, 41)
#         print('x23.shape',x.shape)
        x = self.deconv5_1(x) # (None, 32, 124, 124)
#         print('x24.shape',x.shape)
        x = self.deconv5_2(x) # (None, 32, 124, 124)
#         print('x25.shape',x.shape)
        x = F.pad(x, [-12, -12, -12, -12], mode="constant", value=0) # (None, 32, 100, 100) 
#         print('x26.shape',x.shape)
        x = self.deconv6(x) # (None, 1, 100, 100)
#         print('x27.shape',x.shape)
        return x
        
#can is number of shots
#obs is number of receivers in each shot
class TomoNet(nn.Module):
    def __init__(self, can=30, obs=50):
        
        super(TomoNet, self).__init__()
        self.encoder = Encoder(can, obs)
        self.generator = Generator(can, obs)
        self.decoder = Decoder(can*obs)
    def forward(self, x, p=0.2, training=True):
        
        pre_feature = self.encoder(x)     
       
       
        feature = self.generator(pre_feature)
        
        feature = nn.functional.dropout(feature, p=p, training=training)
        
        d1, d2, d3, out = self.decoder(feature)
        
        
        return feature, d1, d2, d3, out
