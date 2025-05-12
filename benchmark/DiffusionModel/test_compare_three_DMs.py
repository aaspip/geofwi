# Download data from
# https://utexas.box.com/s/pl8yswezkdwenfomgq70yf2r5dl6tha6
# 
# put them in the current folder
# real/initial/pre/pre-sdedit/pre-sgds



import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import sys
import torch
import math

realadd='real/'
iniadd='initial/'
preadd='pre/'  #DM-FWI with MPGD

a=252

add=[323,27,112, 207, 195, 308,  419 , 103,404,  181, 303, 222] 

velmin=1500
velmax=4000

############plot new figures including MPGD, SDEdit and SGDS

preadd3='pre-sdedit/'
preadd6='pre-sgds/'

fig, axes = plt.subplots(8, 6, figsize=(25,25))

for ii,ax in enumerate(axes.flatten()):
	if ii<6:
		pp=ii+1+a
		data=np.load(realadd+str(add[ii])+'.npy')
		im1=ax.imshow(data,vmin=velmin, vmax=velmax)
		
	if ii>=6 and ii<12:
		pp=ii+1+a
		data=np.load(preadd+str(add[ii-6])+'.npy')
		im1=ax.imshow(data,vmin=velmin, vmax=velmax)
		
	if ii>=12 and ii<18:
		pp=ii+1+a
		data=np.load(preadd3+str(add[ii-12])+'.npy')
		im1=ax.imshow(data,vmin=velmin, vmax=velmax)
	if ii>=18 and ii<24:
		pp=ii+1+a
		data=np.load(preadd6+str(add[ii-18])+'.npy')
		im1=ax.imshow(data,vmin=velmin, vmax=velmax)
		
	if ii>=24 and ii<30:
		pp=ii+1+a
		data=np.load(realadd+str(add[ii-18])+'.npy')
		
		im1=ax.imshow(data,vmin=velmin, vmax=velmax)
		
	if ii>=30 and ii<36:
		pp=ii+1+a
		data=np.load(preadd+str(add[ii-24])+'.npy')
		im1=ax.imshow(data,vmin=velmin, vmax=velmax)
		
	if ii>=36 and ii<42:
		pp=ii+1+a
		data=np.load(preadd3+str(add[ii-30])+'.npy')
		im1=ax.imshow(data,vmin=velmin, vmax=velmax)
	if ii>=42 and ii<48:
		pp=ii+1+a
		data=np.load(preadd6+str(add[ii-36])+'.npy')
		im1=ax.imshow(data,vmin=velmin, vmax=velmax)
cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', fraction=0.03, pad=0.04)
cbar.set_label('Vp (m/s)', fontsize=24, fontname="Arial")
cbar.ax.tick_params(labelsize=22)				
plt.text(-950,-785, 'Real', fontsize = 24, fontname="Arial") 
plt.text(-950,-665, 'MPGD', fontsize = 24, fontname="Arial") 
plt.text(-950,-545, 'SDEdit', fontsize = 24, fontname="Arial")
plt.text(-950,-425, 'SGDS', fontsize = 24, fontname="Arial")  
plt.text(-950,-305, 'Real', fontsize = 24, fontname="Arial") 
plt.text(-950,-185, 'MPGD', fontsize = 24, fontname="Arial") 
plt.text(-950,-65, 'SDEdit', fontsize = 24, fontname="Arial") 
plt.text(-950,55, 'SGDS', fontsize = 24, fontname="Arial") 
plt.savefig("geofwiseisdiffusion1.png", dpi=300,bbox_inches='tight')
plt.show()




