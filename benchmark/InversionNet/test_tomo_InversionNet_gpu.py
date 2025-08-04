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

import sys
sys.path.append('../')

import basic_function as func
import CustomDataset as dataset
# import tomography_model_cpu as model

from IPython.core import debugger
debug = debugger.Pdb().set_trace


from functions.inversionnet import InversionNet
# modelnew=InversionNet(dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0)

## Test dimension

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('gpus', metavar='GPUS', help='GPU ID')
# parser.add_argument('modeldir', nargs='?', default=os.getenv('HOME')+'/pylib/demos/')
parser.add_argument('epochs', nargs='?', default=500)
parser.add_argument('batchsize', nargs='?', default=4)
parser.add_argument('lr', nargs='?', default=0.1)
parser.add_argument('workers', nargs='?', default=0)
parser.add_argument('dataset_path', nargs='?', default=os.getenv('HOME')+'/DATALIB/GeoFWI/geofwi_train')
args=parser.parse_args()

#What's the problem: the problem is the mismatch in the model key words
# print(modelnew.state_dict().keys())
# print(modelt['state_dict'].keys())
## change the keys from module.decoder.dsn2.bias to decoder.dsn2.bias
# modelpath='./runs/Mar17_13-42-23_lr1e-5wireless-10-147-80-172.public.utexas.edu/checkpoint.pth.tar'
# modelpath='./runs/Mar17_13-42-23_lr1e-5wireless-10-147-80-172.public.utexas.edu/model_best.pth.tar'
# # modelpath='./runs/Mar03_15-31-48_lr1e-5wireless-10-147-48-47.public.utexas.edu-older/model_best.pth.tar'
# tmp=torch.load(modelpath,weights_only=False)['state_dict']
# tmp2=tmp
# for ii in list(tmp2.keys()):
# 	tmp2[ii[7:]]=tmp2.pop(ii)
# modelnew.load_state_dict(tmp2)

## prediction for training data
# args_train = os.path.join(args.dataset_path, 'train')
# args_valid = os.path.join(args.dataset_path, 'valid')
#         
# 
# train_dataset = dataset.DatasetFolder(args_train, flip=True, norm=True)
# valid_dataset = dataset.DatasetFolder(args_valid, flip=False, norm=True)
#         
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
# valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)
# max_step = args.epochs * len(train_loader)
# 
# print('len(train_loader.dataset)',len(train_loader.dataset))
# print('train_loader.dataset[0][0].shape',train_loader.dataset[0][0].shape)
# print('train_loader.dataset[0][1].shape',train_loader.dataset[0][1].shape)
# 
# ## prediction for training data
# if os.path.isdir('./velcomps-train') == False:  
# 	os.makedirs('./velcomps-train',exist_ok=True)
	
# for ii in range(len(train_loader.dataset)):
# for ii in range(1):
# # 	ii=0 #ith sample
# 	print("ii=%d/%d"%(ii,len(train_loader.dataset)))
# 	observe=train_loader.dataset[ii][0].reshape(1,30,50,1000)
# 	geology=train_loader.dataset[ii][1].reshape(1,1,100,100)
# # 	observe=observe[:,0:5,0:32,0:1000]
# # 	observe=observe[:,0:5,0:32,0:1000].reshape([1,5,1000,32])
# # 	observe=observe.reshape([1,5,1000,300])
# # 	observe=observe[:,:,:,0:70]
# 	predict = modelnew(observe)
# 	predict = predict[:,:,14:114,14:114]
# 	out1 = out1[:,:,14:114,14:114]
# 	out2 = out2[:,:,14:114,14:114]
# 	out3 = out3[:,:,14:114,14:114]

# 	import matplotlib.pyplot as plt
# 	fig = plt.figure(figsize=(12, 4))
# 	plt.subplot(1,3,1)
# 	plt.imshow(geology.detach().numpy().squeeze(),clim=(0, 1),aspect='auto');plt.title('True-%d'%ii);
# 	plt.subplot(1,3,2)
# 	plt.imshow(predict.detach().numpy().squeeze(),clim=(0, 1),aspect='auto');plt.title('Predicted-%d'%ii);
# 	plt.subplot(1,3,3)
# 	plt.imshow(geology.detach().numpy().squeeze()-predict.detach().numpy().squeeze(),clim=(0, 1),aspect='auto');plt.title('Error-%d'%ii);
# 	plt.savefig('velcomps-train/velcomp-%d.png'%ii)
# # 	plt.show()
# 	plt.close()
































#load parameters from main.sh 
# parserWarpper = func.MyArgumentParser()
# parser = parserWarpper.get_parser()
# args = parser.parse_args()
# parameters = [item for item in args.__dict__.items()]    
# print(parameters)

# import argparse
# class MyArgumentParser():
#     def __init__(self, inference=False):
#         self.parser = argparse.ArgumentParser(description='PyTorch Tomo Training')
#         self.parser.add_argument('gpus', metavar='GPUS', help='GPU ID')
#         self.parser.add_argument('dataset_path', metavar='DATASET_PATH', help='path to the dataset')
#         self.parser.add_argument('workers', default=2, type=int, metavar='WORKERS', help='number of dataload worker')
#         if inference:
#             self.parser.add_argument('checkpoint_path', metavar='CHECKPOINT_PATH', help='path to the checkpoint file')
#             self.parser.add_argument('save_path', default='infer', metavar='SAVE_PATH', help='path to the inference results')
#         else:
#             self.parser.add_argument('batchsize', default=4, type=int, metavar='BATCH_SIZE', help='batchsize')
#             self.parser.add_argument('lr', default=0.1, type=float, metavar='LEARNING_RATE', help='learning rate')
#             #self.parser.add_argument('wdecay', default=1e-4, type=float, metavar='WEIGHT_DECAY', help='weight decay')
#             #self.parser.add_argument('momentum', default=0.9, type=float, metavar='MOMENTUM', help='the momentum of SGD learning algorithm')
#             self.parser.add_argument('epochs', default=500, type=int, metavar='EPOCH',help='number of total epochs to run')   
#     def get_parser(self):
#         return self.parser      
            
# args=MyArgumentParser()


# opt_manualSeed = 666
# print("Random Seed: ", opt_manualSeed)
# np.random.seed(opt_manualSeed)
# random.seed(opt_manualSeed)
# torch.manual_seed(opt_manualSeed)
# torch.cuda.manual_seed_all(opt_manualSeed)
# 
# cudnn.benchmark = True
# cudnn.deterministic = False
# # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('gpus', metavar='GPUS', help='GPU ID')
# parser.add_argument('modeldir', nargs='?', default=os.getenv('HOME')+'/pylib/demos/')
parser.add_argument('epochs', nargs='?', default=500)
parser.add_argument('batchsize', nargs='?', default=4)
parser.add_argument('lr', nargs='?', default=0.1)
parser.add_argument('workers', nargs='?', default=0)
parser.add_argument('dataset_path', nargs='?', default=os.getenv('HOME')+'/DATALIB/GeoFWI/geofwi_train')
args=parser.parse_args()
    
#from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self, args):
        self.args = args
        self.date = datetime.now().strftime('%b%d_%H-%M-%S') + '_lr1e-5' + socket.gethostname()
        
        
       
        self.log_dir = os.path.join('./runs/', self.date)
        #self.writer = SummaryWriter(log_dir = self.log_dir)

        self.min_val_loss = 1000000

        args_train = os.path.join(args.dataset_path, 'train')
        args_valid = os.path.join(args.dataset_path, 'valid')
        
        
        #print('args_train',args_train)
        #print('args_valid',args_valid)
        
        train_dataset = dataset.DatasetFolder(args_train, flip=True, norm=True)
       
        valid_dataset = dataset.DatasetFolder(args_valid, flip=False, norm=True)
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)
        max_step = args.epochs * len(self.train_loader)


        
        #print('self.train_loader',self.train_loader)
        #print('self.valid_loader',self.valid_loader)
        
        
        
#         self.model = model.TomoNet()
        self.model = InversionNet()
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        
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
        #losses2 = func.AverageMeter()
        losses3 = func.AverageMeter() 
#         tbar = tqdm(self.train_loader.dataset)
        tbar = tqdm(self.train_loader)
#         tbar=self.train_loader
        self.model.train()
        with torch.enable_grad():
            #print('tbar',tbar)
            #sys.exit()
            for step, (observe, geology, observe_path, geology_path) in enumerate(tbar):
#                 print('step, observe, geology, observe_path, geology_path',step, observe, geology, observe_path, geology_path)
                
                cur_lr = self.scheduler.get_lr()[0]
                
                observe = observe.cuda(non_blocking=True)
                geology = geology.cuda(non_blocking=True)
                
                geo_edge = func.edge_detect(geology)!=0
                
                geo_edge = geo_edge.float()
                
                
                # compute output
#                 feature, out1, out2, out3, predict = self.model(observe, p=0.2, training=True)
                predict = self.model(observe)

                
#                 out1 = out1[:,:,14:114,14:114]
#                 out2 = out2[:,:,14:114,14:114]
#                 out3 = out3[:,:,14:114,14:114]
#                 predict = predict[:,:,14:114,14:114]

#                 edges = torch.cat((geo_edge, torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3)), 0)

                loss1 = self.value(predict, geology)
                #loss2 = edge(out1, geo_edge) + edge(out2, geo_edge) + edge(out3, geo_edge)
                loss3 = 1 - self.ssim(predict, geology)
                loss = loss1 + loss3

                # measure accuracy and record loss
                losses1.update(loss1.item(), observe.size(0)) 
                #losses2.update(loss2.item(), observe.size(0)) 
                losses3.update(loss3.item(), observe.size(0))
                losses.update(loss.item(), observe.size(0))  

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                tbar.set_description('Train [{0}] Loss {loss.val:.5f} {loss.avg:.5f} Lr {lr:.5f} Min_val {min:.5f}'.format(epoch, loss=losses, lr=cur_lr, min=self.min_val_loss))
                # self.writer.add_scalar('data/lr', cur_lr, epoch*len(self.train_loader)+step)
                # self.writer.add_scalar('data/loss_train', loss.item(), epoch*len(self.train_loader)+step)
                # if step % (100) == 0:
                    # threadlist.append(threading.Thread(target=func.to_tensorboard, args=(lock, self.writer, torch.cat((geology, predict, edges),0).cpu(), epoch*len(self.train_loader)+step, 'img/img_train', self.args.batchsize)))
                    # threadlist[-1].start()

            # #print('main thread finished, waiting for IO threads...')
            # #for thread in threadlist:
            # #    thread.join()
            # self.writer.add_scalar('data/Loss1_Train', losses1.avg, epoch)        
            # #self.writer.add_scalar('data/Loss2_Train', losses2.avg, epoch)
            # self.writer.add_scalar('data/Loss3_Train', losses3.avg, epoch)        
            # self.writer.add_scalar('data/Loss_Train', losses.avg, epoch)

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

                observe = observe.cuda(non_blocking=True)
                geology = geology.cuda(non_blocking=True)

                geo_edge = func.edge_detect(geology)!=0
                geo_edge = geo_edge.float()

                # compute output
#                 feature, out1, out2, out3, predict = self.model(observe, p=0, training=False)
                predict = self.model(observe)

#                 out1 = out1[:,:,14:114,14:114]
#                 out2 = out2[:,:,14:114,14:114]
#                 out3 = out3[:,:,14:114,14:114]
#                 predict = predict[:,:,14:114,14:114]

#                 edges = torch.cat((geo_edge, torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3)), 0)

                loss1 = self.value(predict, geology)
                #loss2 = edge(out1, geo_edge) + edge(out2, geo_edge) + edge(out3, geo_edge)
                loss3 = 1 - self.ssim(predict, geology)
                loss = loss1 + loss3

                # measure accuracy and record loss
                losses1.update(loss1.item(), observe.size(0)) 
                #losses2.update(loss2.item(), observe.size(0)) 
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
    

    
    
    
    
    
    
    
    
    
    
    
