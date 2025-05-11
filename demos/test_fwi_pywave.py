from pywave import pfwi #pip install git+https://github.com/chenyk1990/pywave
import numpy as np
import matplotlib.pyplot as plt
import os
from pyseistr import binread,smooth #pip install git+https://github.com/aaspip/pyseistr

# DataPath
#https://github.com/chenyk1990/cykd2/blob/master/various/cyksmall/marmvel.bin

## load velocity
# marmvel=binread(os.getenv('HOME')+'/chenyk.data2/various/cyksmall/marmvel.bin',n1=751,n2=2301)

# marmvel=binread('marmvel.bin',n1=751,n2=2301)
# marmvel=marmvel*0.001;
# marmvel=marmvel[0:240,1250:2251];
# marmvel=marmvel[::2,::3]*1.8-1.2;
# vel=np.zeros([130,334]);vel[10:,:]=marmvel;vel[0:10,:]=np.multiply(np.ones([10,1]),marmvel[0,:]);

from geofwi import get_vel
vel=get_vel(layer=5,mode='salt',datapath='./data',indx=0) #turn into the last sample of the last type
import matplotlib.pyplot as plt
plt.imshow(vel,cmap=plt.jet());plt.colorbar();
plt.show()


	
## obtain start velocity
vel_s=np.concatenate([vel[0:14],1./smooth(1/vel[14:,:],[20,20,1])],axis=0)
q=np.ones(vel.shape)*10000;
plt.subplot(2,1,1);
plt.imshow(vel);
plt.subplot(2,1,2);
plt.imshow(vel_s);
plt.show()

## wavelet
from pyseistr import ricker
trace=np.zeros(3001);trace[99]=1000000;dt=0.001;
wav,tw=ricker(10,dt,0.2)
wavelet=np.convolve(trace,wav,mode='same');
plt.plot(wavelet);plt.show()

## src
# [nz,nx]=vel.shape
# nt=len(wavelet)
# src=np.zeros([nz,nx,nt,3]);
# s1=np.array([[110,120,117,120,112,115],[100,110,165,205,220,260],[100,300,500,700,900,1100]]);
# s2=np.array([[115,111,120,122,118,115],[95,120,170,200,230,255],[100,300,500,700,900,1100]]);
# s3=np.array([[112,118,111,120,116,110],[90,110,145,180,200,240],[100,300,500,700,900,1100]]);
# nfrac=s1.shape[1]; #number of fractions
# for ii in range(nfrac):
# 	src[s1[0,ii],s1[1,ii],s1[2,ii],0] = 1e6;
# 	src[s2[0,ii],s2[1,ii],s2[2,ii],1] = 1e6;
# 	src[s3[0,ii],s3[1,ii],s3[2,ii],2] = 1e6;
# 	
# for ii in range(nz):
# 	for jj in range(nx):
# 		for kk in range(3):
# 			src[ii,jj,:,kk]=np.convolve(src[ii,jj,:,kk],wav,mode='same');
# plt.imshow(np.sum(np.sum(src,3),2));
# plt.show()

## plot source
# plt.subplot(1,2,1);
# plt.plot(src[110,100,:,0]);
# plt.subplot(1,2,2);
# plt.plot(src[115,260,:,0]);
# plt.show()

## generate data
mypar={'nz':130, 'nx':334, 'dz': 0.008, 'dx': 0.012, 'oz': 0, 'ox': 0, 'ns': 20, 'ds': 0.2,
		'nt': 3001, 'dt': 0.001, 'ot': 0, 'nb':60, 'coef': 0.005, 'acqui_type': 1, 
		'inv': 0, 'waterz': -30, 'onlysrc': 0, 'onlyvel': 1, 'conv_error': 0.01, 'niter': 30}

## Modeling part
# data,vinv,grad,src,mwt=pfwi(vel,q,wavelet,src,data=None,mode=4,media=1,inv=0,verb=1,par=mypar);
# np.save('datas',data);
# plt.subplot(1,3,1);
# plt.imshow(data[:,:,0],aspect='auto');
# plt.subplot(1,3,2);
# plt.imshow(data[:,:,1],aspect='auto');
# plt.subplot(1,3,3);
# plt.imshow(data[:,:,2],aspect='auto');
# plt.show()

## Source inversion part
mypar['inv']=True;
mypar['onlysrc']=True;
mypar['niter']=30;
# data,vinv,grad,src,mwt=pfwi(vel,q,wavelet,src=None,data=None,mode=1,media=1,inv=0,verb=1,par=mypar);
# np.save('fwi-datas-%d.npy'%mypar['ns'],data)

data=np.load('fwi-datas-%d.npy'%mypar['ns'])

par=mypar
plt.figure(figsize=(12, 8))
plt.subplot(1,6,1);
plt.imshow(data[:,:,0],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Shot 1"); plt.ylabel("Time (s)"); plt.xlabel("Receiver (km)"); 
plt.subplot(1,6,2);
plt.imshow(data[:,:,4],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Shot 5"); plt.gca().set_yticks([]); plt.xlabel("Receiver (km)"); 
plt.subplot(1,6,3);
plt.imshow(data[:,:,8],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Shot 9"); plt.gca().set_yticks([]); plt.xlabel("Receiver (km)"); 

plt.subplot(1,6,4);
plt.imshow(data[:,:,12],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Shot 13"); plt.gca().set_yticks([]);plt.xlabel("Receiver (km)"); 
plt.subplot(1,6,5);
plt.imshow(data[:,:,15],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Shot 16"); plt.gca().set_yticks([]); plt.xlabel("Receiver (km)"); 
plt.subplot(1,6,6);
plt.imshow(data[:,:,19],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Shot 20"); plt.gca().set_yticks([]); plt.xlabel("Receiver (km)"); 

plt.savefig(fname='test_pfwi_vel2d_data.png',format='png',dpi=300)

plt.show()