import os
import shutil
import argparse
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import scipy.io as sio
from skimage import feature

from IPython.core import debugger
debug = debugger.Pdb().set_trace

def normalize(data):
    batch = data.size(0)
    for i in range(batch):
        data[i] =  (data[i] - data[i].min()) / (data[i].max() - data[i].min())
    return data


def colortojet(img):
    cmap=plt.get_cmap('jet')
    img = img.numpy()
    img_jet = cmap(img)[:,:,:,0:3]
    img_out = torch.from_numpy(img_jet).transpose(0,3).squeeze(-1).float()
    return img_out

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]
    return lr


def save_checkpoint(state, directory, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    #directory = "./runs/{}/".format(date)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.minimum = 1000000
        self.maximum = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.minimum = val if self.minimum > val else self.minimum
        self.maximum = val if self.maximum < val else self.maximum
        

def to_tensorboard(lock, writer, img, iter_num, tag='Obs&Pre', nrow=8, norm=False):
    lock.acquire()
    show_img = vutils.make_grid(img, nrow=nrow, padding=2, normalize=norm, scale_each=norm, pad_value=0)
    writer.add_image(tag, show_img, iter_num)           
    lock.release()

    
def visualize_img(input, target, prob):
    img = torch.zeros_like(input, device='cpu', requires_grad=False)
    img.copy_(input)
    img = img.repeat(1,3,1,1)
    img = F.interpolate(img, size=200)
    batchsize = input.size(0)
    for i in range(batchsize):
        img_pil = transforms.functional.to_pil_image(norm(img[i]))
        img_draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype('Arial.ttf', 20)
        img_draw.text((0, 0),str(target[i].item())+'/'+str("%.4f" % prob[i].item()),(255,0,0),font=font)
        img[i] = transforms.functional.to_tensor(img_pil)
    return img    
    

def save_img(lock, img, path, nrow=8, norm=False):
    lock.acquire()
    batch = img.size(0)
    for i in range(batch):
        tpath = path[i].replace('valid','valid_predict').replace('test','test_predict').replace('train','train_predict').replace('mat','png')
        File_Path = tpath[0:tpath.rfind('/')]
        if not os.path.exists(File_Path):
            os.makedirs(File_Path)
        vutils.save_image(img[i], tpath, nrow=nrow, padding=0, normalize=norm, scale_each=norm)
    lock.release()


def save_mat(data, path, postfix):
    tpath = path[0].replace('valid','valid_predict_mat').replace('test','test_predict_mat').replace('train','train_predict_mat')
    File_Path = tpath[0:tpath.rfind('/')]
    if not os.path.exists(File_Path):
        os.makedirs(File_Path)
    filename = tpath[0:tpath.rfind('.')] + '_' + postfix + '.mat'
    sio.savemat(filename, {postfix: data})


def save_features(data, path, postfix):
    tpath = path[0].replace('valid','valid_predict_mat').replace('test','test_predict_mat').replace('train','train_predict_mat')
    File_Path = tpath[0:tpath.rfind('/')]
    if not os.path.exists(File_Path):
        os.makedirs(File_Path)
    filename = tpath[0:tpath.rfind('.')] + '_' + postfix + '.png'
    vutils.save_image(data[:,1:-1,1:-1].unsqueeze(1), filename, nrow=20, padding=0, normalize=True, scale_each=False)


def save_feature(lock, img, feature, path):
    lock.acquire()
    batch = img.size(0)
    for i in range(batch):
        img_path = path[i].replace('valid','valid_predict').replace('test','test_predict').replace('train','train_predict').replace('mat','png')
        feat_path = path[i].replace('valid','valid_predict').replace('test','test_predict').replace('train','train_predict').replace('.mat','_feat.png')
        File_Path = img_path[0:img_path.rfind('/')]
        if not os.path.exists(File_Path):
            os.makedirs(File_Path)
        vutils.save_image(img[i], img_path, nrow=1, padding=0, normalize=False, scale_each=False)
        vutils.save_image(feature[i].unsqueeze(1), feat_path, nrow=20, padding=0, normalize=True, scale_each=True)
    lock.release()


def edge_detect(x):   
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    h_tv = torch.pow((x[:,:,1:,:-1]-x[:,:,:h_x-1,:-1]),2).sum(1, True)
    w_tv = torch.pow((x[:,:,:-1,1:]-x[:,:,:-1,:w_x-1]),2).sum(1, True)
    return F.pad(h_tv + w_tv, (0, 1, 0, 1), 'replicate')    
 

def canny(im):
    edge = feature.canny(im.squeeze().cpu().numpy()).astype(np.float16)
    return torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).cuda()

class SegLoss(nn.Module):
    def __init__(self, ignore_label=-100):
        super(SegLoss, self).__init__()
        self.obj = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
    def forward(self, pred, label):        
        loss = self.obj(pred, label)
        return loss    
    

class EdgeLoss(nn.Module):
    def __init__(self, mode='normal'):
        super(EdgeLoss, self).__init__()
        self.mode = mode
    def forward(self, pred, target):
        #pred_edge = torch.tanh(edge_detect(pred))
        #target = nn.functional.interpolate(target, pred_edge.size()[2:3])
        beta = 1 - torch.mean(target)
        weights = 1 - beta + (2 * beta - 1) * target
        if self.mode == 'normal':
            loss = F.binary_cross_entropy(pred, target, weights)
        else:
            loss = F.binary_cross_entropy_with_logits(pred, target, weights)
        #loss = nn.functional.mse_loss(pred_edge, target)
        return loss


class EdgeLossVanila(nn.Module):
    def __init__(self, mode='normal'):
        super(EdgeLossVanila, self).__init__()
        self.mode = mode
    def forward(self, pred, target):
        if self.mode == 'normal':
            loss = F.binary_cross_entropy(pred, target)
        else:
            loss = F.binary_cross_entropy_with_logits(pred, target)
        return loss


class SoftEdgeLoss(nn.Module):
    def __init__(self, mode='normal'):
        super(SoftEdgeLoss, self).__init__()
        self.mode = mode
    def forward(self, pred, target):
        kernel = torch.from_numpy(gkern(kernlen=5)).float().unsqueeze(0).unsqueeze(0).cuda()
        pred = nn.functional.pad(pred, (2,2,2,2), 'replicate')
        pred = F.conv2d(pred, kernel)
        # target = (target-target.min())/(target.max()-target.min())
        beta = 1 - torch.mean(target)
        weights = 1 - beta + (2 * beta - 1) * target
        if self.mode == 'normal':
            loss = F.binary_cross_entropy(pred, target, weights)
        else:
            loss = F.binary_cross_entropy_with_logits(pred, target, weights)
        return loss


class SoftIOU(nn.Module):
    def __init__(self):
        super(SoftIOU, self).__init__()
    def forward(self, pred, target):
        kernel = torch.from_numpy(gkern(kernlen=7)).float().unsqueeze(0).unsqueeze(0).cuda()
        pred_soft = nn.functional.pad(pred, (3,3,3,3), 'replicate')
        pred_soft = F.conv2d(pred_soft, kernel)
        pred_soft = (pred_soft - pred_soft.min()) / (pred_soft.max() - pred_soft.min())
        TP = torch.sum(pred_soft[target == 1])
        if torch.sum(pred) <= 0 or torch.isnan(TP) or TP == 0:
            #torch.sum(torch.isnan(TP))>0:
            #debug()
            Fmeasure = torch.zeros(1)  
        else:
            #TP = torch.sum(pred_soft[target == 1])
            Precise = TP / torch.sum(pred)
            Recall = TP / torch.sum(target)
            Fmeasure = 2 * (Precise * Recall) / (Precise + Recall) 
        if torch.isnan(Fmeasure):
            debug()    
        return Fmeasure #, Precise, Recall


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def gaussian(window_size, sigma):
    gauss = torch.exp(torch.Tensor([-(x - window_size//2)**2/float(2*sigma**2) for x in range(window_size)]))
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window.to(img1.device), window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)


class MyArgumentParser():
    def __init__(self, inference=False):
        self.parser = argparse.ArgumentParser(description='PyTorch Tomo Training')
        self.parser.add_argument('gpus', metavar='GPUS', help='GPU ID')
        self.parser.add_argument('dataset_path', metavar='DATASET_PATH', help='path to the dataset')
        self.parser.add_argument('workers', default=2, type=int, metavar='WORKERS', help='number of dataload worker')
        if inference:
            self.parser.add_argument('checkpoint_path', metavar='CHECKPOINT_PATH', help='path to the checkpoint file')
            self.parser.add_argument('save_path', default='infer', metavar='SAVE_PATH', help='path to the inference results')
        else:
            self.parser.add_argument('batchsize', default=4, type=int, metavar='BATCH_SIZE', help='batchsize')
            self.parser.add_argument('lr', default=0.1, type=float, metavar='LEARNING_RATE', help='learning rate')
            #self.parser.add_argument('wdecay', default=1e-4, type=float, metavar='WEIGHT_DECAY', help='weight decay')
            #self.parser.add_argument('momentum', default=0.9, type=float, metavar='MOMENTUM', help='the momentum of SGD learning algorithm')
            self.parser.add_argument('epochs', default=500, type=int, metavar='EPOCH',help='number of total epochs to run')   
    def get_parser(self):
        return self.parser
