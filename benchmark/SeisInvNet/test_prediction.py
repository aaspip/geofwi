import os
import sys
import time
import random
import socket
import threading
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torchnet as tnt
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, Function
from tqdm import tqdm

import geofwi.train as func
import geofwi.data as dataset
import geofwi.seisinvnet as model

from IPython.core import debugger
debug = debugger.Pdb().set_trace

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('epochs', nargs='?', default=500)
parser.add_argument('batchsize', nargs='?', default=4)
parser.add_argument('lr', nargs='?', default=0.1)
parser.add_argument('workers', nargs='?', default=0)
parser.add_argument('dataset_path', nargs='?', default=os.getenv('HOME')+'/DATALIB/GeoFWI/geofwi_train') #change this to other directory where you store the shot gathers
args=parser.parse_args()

class Trainer():
    def __init__(self, args):
        self.args = args
        self.date = datetime.now().strftime('%b%d_%H-%M-%S') + '_lr1e-5' + socket.gethostname()
       
        self.log_dir = os.path.join('./runs/', self.date)

        self.min_val_loss = 1000000

        args_train = os.path.join(args.dataset_path, 'train')
        args_valid = os.path.join(args.dataset_path, 'valid')
        
        train_dataset = dataset.DatasetFolder(args_train, flip=True, norm=True)
       
        valid_dataset = dataset.DatasetFolder(args_valid, flip=False, norm=True)
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)
        max_step = args.epochs * len(self.train_loader)
        
        self.model = model.TomoNet()
        self.model = torch.nn.DataParallel(self.model)
#         self.model = self.model.cuda() #no need for GPU for prediction only 

        #set optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 0.1 * (1.0-float(step)/max_step)**0.9)
        self.value = nn.MSELoss()
        self.edge = func.EdgeLoss('logits')
        self.ssim = func.MSSSIM()

    def train(self, epoch):
        print('epoch',epoch)
        lock = threading.Lock()
        threadlist = []
        losses = func.AverageMeter()
        losses1 = func.AverageMeter()
        losses3 = func.AverageMeter() 
        tbar = tqdm(self.train_loader)
        self.model.train()
        with torch.enable_grad():
            for step, (observe, geology, observe_path, geology_path) in enumerate(tbar):
                
                cur_lr = self.scheduler.get_lr()[0]
                
                geo_edge = func.edge_detect(geology)!=0
                
                geo_edge = geo_edge.float()
                
                # compute output
                feature, out1, out2, out3, predict = self.model(observe, p=0.2, training=True)
                out1 = out1[:,:,14:114,14:114]
                out2 = out2[:,:,14:114,14:114]
                out3 = out3[:,:,14:114,14:114]
                predict = predict[:,:,14:114,14:114]

                edges = torch.cat((geo_edge, torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3)), 0)

                loss1 = self.value(predict, geology)
                loss3 = 1 - self.ssim(predict, geology)
                loss = loss1 + loss3

                # measure accuracy and record loss
                losses1.update(loss1.item(), observe.size(0)) 
                losses3.update(loss3.item(), observe.size(0))
                losses.update(loss.item(), observe.size(0))  

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                tbar.set_description('Train [{0}] Loss {loss.val:.5f} {loss.avg:.5f} Lr {lr:.5f} Min_val {min:.5f}'.format(epoch, loss=losses, lr=cur_lr, min=self.min_val_loss))

    def validate(self, epoch):
       
       
        lock = threading.Lock()
        threadlist = []
        losses = func.AverageMeter()
        losses1 = func.AverageMeter()
        #losses2 = func.AverageMeter()
        losses3 = func.AverageMeter()
        tbar = tqdm(self.valid_loader)
        self.model.eval()
        with torch.no_grad():
            for step, (observe, geology, observe_path, geology_path) in enumerate(tbar):

                geo_edge = func.edge_detect(geology)!=0
                geo_edge = geo_edge.float()

                # compute output
                feature, out1, out2, out3, predict = self.model(observe, p=0, training=False)

                out1 = out1[:,:,14:114,14:114]
                out2 = out2[:,:,14:114,14:114]
                out3 = out3[:,:,14:114,14:114]
                predict = predict[:,:,14:114,14:114]

                loss1 = self.value(predict, geology)
                loss3 = 1 - self.ssim(predict, geology)
                loss = loss1 + loss3

                # measure accuracy and record loss
                losses1.update(loss1.item(), observe.size(0)) 
                losses3.update(loss3.item(), observe.size(0))
                losses.update(loss.item(), observe.size(0))  
               
                tbar.set_description('Valid [{0}] Loss {loss.val:.5f} {loss.avg:.5f}'.format(epoch, loss=losses))
            
            return losses.avg


trainer = Trainer(args)

# trainer2 = Trainer(args)
import tomography_model_cpu as model
modelnew=model.TomoNet()
#What's the problem? The problem is the mismatch in the model key words
# print(modelnew.state_dict().keys())
# print(modelt['state_dict'].keys())
## change the keys from module.decoder.dsn2.bias to decoder.dsn2.bias

#This model is the currently best model, trained from 40000 samples, validated on 5000 samples, tested on the rest samples
#downloadable from https://utexas.box.com/s/6ak8omw3rlvatdnh51ngig1n94e02mss
modelpath='SeisInvNet_model_40000_400epoches.pth' 

tmp=torch.load(modelpath,weights_only=False,map_location=torch.device('cpu'))['state_dict']
tmp2=tmp
for ii in list(tmp2.keys()):
	tmp2[ii[7:]]=tmp2.pop(ii)
modelnew.load_state_dict(tmp2)

## prediction for training data
args_train = os.path.join(args.dataset_path, 'train')
args_valid = os.path.join(args.dataset_path, 'valid')
args_test = os.path.join(args.dataset_path, 'test')

print('args_train',args_train)
print('args_valid',args_valid)
print('args_test',args_test)

train_dataset = dataset.DatasetFolder(args_train, flip=True, norm=True)
valid_dataset = dataset.DatasetFolder(args_valid, flip=False, norm=True)
test_dataset = dataset.DatasetFolder(args_test, flip=False, norm=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)
max_step = args.epochs * len(train_loader)

print('len(train_loader.dataset)',len(train_loader.dataset))
print('train_loader.dataset[0][0].shape',train_loader.dataset[0][0].shape)
print('train_loader.dataset[0][1].shape',train_loader.dataset[0][1].shape)
print('test_loader.dataset[0][1].shape',test_loader.dataset[0][1].shape)

## prediction for training data
if os.path.isdir('./velcomps-train') == False:  
	os.makedirs('./velcomps-train',exist_ok=True)
	
for ii in range(len(train_loader.dataset)):
	print("ii=%d/%d"%(ii,len(train_loader.dataset)))
	observe=train_loader.dataset[ii][0].reshape(1,30,50,1000)
	geology=train_loader.dataset[ii][1].reshape(1,1,100,100)

	feature, out1, out2, out3, predict = modelnew(observe, p=0.2, training=True)
	predict = predict[:,:,14:114,14:114]
	out1 = out1[:,:,14:114,14:114]
	out2 = out2[:,:,14:114,14:114]
	out3 = out3[:,:,14:114,14:114]

	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(12, 4))
	plt.subplot(1,3,1)
	plt.imshow(geology.detach().numpy().squeeze(),clim=(0, 1),aspect='auto');plt.title('True-%d'%ii);
	plt.subplot(1,3,2)
	plt.imshow(predict.detach().numpy().squeeze(),clim=(0, 1),aspect='auto');plt.title('Predicted-%d'%ii);
	plt.subplot(1,3,3)
	plt.imshow(geology.detach().numpy().squeeze()-predict.detach().numpy().squeeze(),clim=(0, 1),aspect='auto');plt.title('Error-%d'%ii);
	plt.savefig('velcomps-train/velcomp-%d.png'%ii)
# 	plt.show()
	plt.close()

## prediction for validation data
if os.path.isdir('./velcomps-valid') == False:  
	os.makedirs('./velcomps-valid',exist_ok=True)
for ii in range(len(valid_loader.dataset)):
	print("ii=%d/%d"%(ii,len(valid_loader.dataset)))
	
	observe=valid_loader.dataset[ii][0].reshape(1,30,50,1000)
	geology=valid_loader.dataset[ii][1].reshape(1,1,100,100)

	feature, out1, out2, out3, predict = modelnew(observe, p=0.2, training=True)
	predict = predict[:,:,14:114,14:114]
	out1 = out1[:,:,14:114,14:114]
	out2 = out2[:,:,14:114,14:114]
	out3 = out3[:,:,14:114,14:114]

	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(12, 4))
	plt.subplot(1,3,1)
	plt.imshow(geology.detach().numpy().squeeze(),clim=(0, 1),aspect='auto');plt.title('True-%d'%ii);
	plt.subplot(1,3,2)
	plt.imshow(predict.detach().numpy().squeeze(),clim=(0, 1),aspect='auto');plt.title('Predicted-%d'%ii);
	plt.subplot(1,3,3)
	plt.imshow(geology.detach().numpy().squeeze()-predict.detach().numpy().squeeze(),clim=(0, 1),aspect='auto');plt.title('Error-%d'%ii);
	plt.savefig('velcomps-valid/velcomp-%d.png'%ii)
# 	plt.show()
	plt.close()

if os.path.isdir('./velcomps-test') == False:  
	os.makedirs('./velcomps-test',exist_ok=True)
for ii in range(len(test_loader.dataset)):
	print("ii=%d/%d"%(ii,len(test_loader.dataset)))
	
	observe=test_loader.dataset[ii][0].reshape(1,30,50,1000)
	geology=test_loader.dataset[ii][1].reshape(1,1,100,100)

	feature, out1, out2, out3, predict = modelnew(observe, p=0.2, training=True)
	predict = predict[:,:,14:114,14:114]
	out1 = out1[:,:,14:114,14:114]
	out2 = out2[:,:,14:114,14:114]
	out3 = out3[:,:,14:114,14:114]

	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(12, 4))
	plt.subplot(1,3,1)
	plt.imshow(geology.detach().numpy().squeeze(),clim=(0, 1),aspect='auto');plt.title('True-%d'%ii);
	plt.subplot(1,3,2)
	plt.imshow(predict.detach().numpy().squeeze(),clim=(0, 1),aspect='auto');plt.title('Predicted-%d'%ii);
	plt.subplot(1,3,3)
	plt.imshow(geology.detach().numpy().squeeze()-predict.detach().numpy().squeeze(),clim=(0, 1),aspect='auto');plt.title('Error-%d'%ii);
	plt.savefig('velcomps-test/velcomp-%d.png'%ii)
# 	plt.show()
	plt.close()

