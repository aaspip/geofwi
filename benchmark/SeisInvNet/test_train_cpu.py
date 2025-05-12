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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('epochs', nargs='?', default=500)
parser.add_argument('batchsize', nargs='?', default=4)
parser.add_argument('lr', nargs='?', default=0.1)
parser.add_argument('workers', nargs='?', default=0)
parser.add_argument('dataset_path', nargs='?', default=os.getenv('HOME')+'/DATALIB/GeoFWI/geofwi_train')
#change this to other directory where you store the shot gathers
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

for epoch in range(args.epochs):

    # train and validate
    trainer.train(epoch)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        val_loss = trainer.validate(epoch)
    
    # save checkpoint
    is_best = val_loss < trainer.min_val_loss
    trainer.min_val_loss = val_loss if is_best else trainer.min_val_loss
    
                   
    func.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': trainer.model.state_dict(),
        'min_val_loss': val_loss,
        'optimizer': trainer.optimizer.state_dict(),
    }, trainer.log_dir, is_best)
    
    
    
    
    
    
    
    
    
    
    
    
