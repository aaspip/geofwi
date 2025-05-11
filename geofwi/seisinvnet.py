import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import sys

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
    def __init__(self, num_shots):
        super(Neighbor, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2)
        self.norm1 = nn.InstanceNorm3d(num_shots)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=5, padding=2)
        self.norm2 = nn.InstanceNorm3d(num_shots)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=5, padding=2)
        self.norm3 = nn.InstanceNorm3d(num_shots)
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
    def __init__(self, num_shots, num_receivers):
        super(Encoder, self).__init__()
        self.num_shots = num_shots
        self.num_receivers = num_receivers

        self.neig = Neighbor(num_shots)
        self.glob = Global()

        self.relu = nn.ReLU(inplace = True)
        c_ind = torch.eye(num_shots,num_shots).repeat(1,num_receivers).reshape(-1).view(-1,num_shots)
        o_ind = torch.eye(num_receivers,num_receivers).repeat(num_shots,1)
        self.ind = torch.cat((c_ind,o_ind),1).unsqueeze(0).view(1,-1,num_shots+num_receivers)
        
    def forward(self, x):
        b,c,h,w = x.shape
        neig_ = self.neig(x) #why self.neighbor->self.neig works
        glob_ = self.glob(x).squeeze()
        glob_ = glob_.view(b,self.num_shots,1,-1).repeat(1,1,self.num_receivers,1).view(b,self.num_shots*self.num_receivers,-1)
        neig_ = neig_.view(b,-1,w)
#         y = torch.cat((glob_,self.ind.repeat(b,1,1).cuda(),neig_),2)
        y = torch.cat((glob_,self.ind.repeat(b,1,1),neig_),2)
        return y
        

class Generator(nn.Module):
    def __init__(self, num_shots=30, num_receivers=50):
        super(Generator, self).__init__()
        self.num = num_shots * num_receivers
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

#num_shots is number of shots
#num_receivers is number of receivers in each shot
class TomoNet(nn.Module):
    def __init__(self, num_shots=30, num_receivers=50):
        
        super(TomoNet, self).__init__()
        self.encoder = Encoder(num_shots, num_receivers)
        self.generator = Generator(num_shots, num_receivers)
        self.decoder = Decoder(num_shots*num_receivers)
    def forward(self, x, p=0.2, training=True):
        
        pre_feature = self.encoder(x)     
       
       
        feature = self.generator(pre_feature)
        
        feature = nn.functional.dropout(feature, p=p, training=training)
        
        d1, d2, d3, out = self.decoder(feature)
        
        
        return feature, d1, d2, d3, out
