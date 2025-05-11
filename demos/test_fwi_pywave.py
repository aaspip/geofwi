from pywave import pfwi #pip install git+https://github.com/chenyk1990/pywave
from pyseistr import smooth #pip install git+https://github.com/aaspip/pyseistr

import numpy as np
import matplotlib.pyplot as plt
import os

## Extract on velocity sample from GeoFWI database
from geofwi import get_vel
vel=get_vel(layer=10,mode='fault',datapath='../data',indx=0)/1000.0 #turn into the last sample of the last type
import matplotlib.pyplot as plt
plt.imshow(vel,cmap=plt.jet());plt.colorbar();
plt.show()
	
## obtain start velocity
vel_s = smooth(vel,[4,10,1]);
q=np.ones(vel.shape)*10000;
plt.subplot(2,1,1);
plt.imshow(vel, clim=[1.5,4.5]);
plt.subplot(2,1,2);
plt.imshow(vel_s, clim=[1.5,4.5]);
plt.show()

## wavelet
from pyseistr import ricker
trace=np.zeros(3001);trace[99]=1000000;dt=0.001;
wav,tw=ricker(10,dt,0.2)
wavelet=np.convolve(trace,wav,mode='same');
plt.plot(wavelet);plt.show()

## generate data
mypar={'nz':100, 'nx':100, 'dz': 0.01, 'dx': 0.01, 'oz': 0, 'ox': 0, 'ns': 20, 'ds': 0.05,
		'nt': 1501, 'dt': 0.001, 'ot': 0, 'nb':60, 'coef': 0.005, 'acqui_type': 1, 
		'inv': 0, 'waterz': -30, 'onlysrc': True, 'onlyvel': 1, 'conv_error': 0.01, 'niter': 5}

mypar['inv']=True;
mypar['onlysrc']=True;
## Generate data from ground-truth velocity for FWI
data,tmp,tmp,tmp,tmp=pfwi(vel,q,wavelet,src=None,data=None,mode=1,media=1,inv=0,verb=1,par=mypar);
np.save('fwi-datas-%d.npy'%mypar['ns'],data)

## FWI
mypar['niter']=30;
data=np.load('fwi-datas-%d.npy'%mypar['ns'])
tmp,vinv,grad,tmp,tmp=pfwi(vel_s,q,wavelet,src=None,data=data,mode=2,media=1,inv=0,verb=0,par=mypar);

## Visualize data 
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

plt.savefig(fname='test_fwi_pywave_data.png',format='png',dpi=300)
plt.show()


## Visualize FWI results
par=mypar;
plt.figure(figsize=(8, 10))
plt.subplot(4,1,1);
plt.imshow(vel,cmap=plt.jet(),aspect='auto',clim=(1.5, 4.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]); 
plt.title("Ground truth"); plt.ylabel("Depth (km)"); plt.gca().set_xticks([]);
plt.subplot(4,1,2);
plt.imshow(vel_s,aspect='auto',clim=(1.5, 4.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]); 
plt.title("Initial model"); plt.ylabel("Depth (km)"); plt.gca().set_xticks([]);
plt.subplot(4,1,3);
plt.imshow(vinv[:,:,int(par['niter']/2)],aspect='auto',clim=(1.5, 4.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]);
plt.title("%d Iterations"%int(par['niter']/2));plt.ylabel("Depth (km)"); plt.gca().set_xticks([]);
plt.subplot(4,1,4);
plt.imshow(vinv[:,:,par['niter']-1],aspect='auto',clim=(1.5, 4.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]);
plt.title("%d Iterations"%par['niter']); plt.ylabel("Depth (km)"); plt.xlabel("Lateral (km)"); 
plt.savefig(fname='test_fwi_pywave_vel-%d.png'%mypar['niter'],format='png',dpi=300)
plt.show()




