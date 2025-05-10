import numpy as np
import matplotlib.pyplot as plt
import os

## Define GeoFWI datapath
GeoFWI_path=os.getenv('HOME')+'/DATALIB/GeoFWI/geofwi.npy' #define your own path
#GeoFWI_path = './geofwi.npy' #for example

sizes=np.load('../data/geofwi-size-layer-fault-salt-1-10.npy') #if you run this script in demos/

geofwi=np.load(GeoFWI_path)
nsample=geofwi.shape[0]

np.random.seed(20232425)
inds=np.arange(49500)
np.random.shuffle(inds)

fig=plt.figure(figsize=(16, 16))
for ii in range(25):
	plt.subplot(5,5,ii+1)
	plt.imshow(geofwi[inds[ii],:,:],clim=[1500,4000]);
plt.colorbar(orientation='horizontal',cax=fig.add_axes([0.37,0.07,0.3,0.01]),shrink=1,label='Velocity (m/s)');
plt.savefig('geofwi-samples-random.png',dpi=300)
plt.show()

fig=plt.figure(figsize=(16, 16))
for ii in range(25):
	plt.subplot(5,5,ii+1)
	plt.imshow(geofwi[ii,:,:],clim=[1500,4000]);
plt.colorbar(orientation='horizontal',cax=fig.add_axes([0.37,0.07,0.3,0.01]),shrink=1,label='Velocity (m/s)');
plt.savefig('geofwi-samples-increasing.png',dpi=300)
plt.show()



fig=plt.figure(figsize=(16, 16))
for ii in range(25):
	plt.subplot(5,5,ii+1)
	plt.imshow(geofwi[nsample-ii-1,:,:],clim=[1500,4000]);
plt.colorbar(orientation='horizontal',cax=fig.add_axes([0.37,0.07,0.3,0.01]),shrink=1,label='Velocity (m/s)');
plt.savefig('geofwi-samples-decreasing.png',dpi=300)
plt.show()


fig=plt.figure(figsize=(16, 16))
for ii in range(25):
	plt.subplot(5,5,ii+1)
	plt.imshow(geofwi[nsample-sizes[-1]-1-ii,:,:],clim=[1500,4000]);
plt.colorbar(orientation='horizontal',cax=fig.add_axes([0.37,0.07,0.3,0.01]),shrink=1,label='Velocity (m/s)');
plt.savefig('geofwi-samples-faults.png',dpi=300)

plt.show()


