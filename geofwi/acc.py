# import os

def acc_occurence(yesno,aefapath='./'):
	'''
	acc_occurence: calculate occurence accuracy
	
	EXAMPLE
	
	from pylib.io import asciiread
	
	yesnoall=asciiread('yesnoall_cyk_new.txt');
	yesnoall=[float(ii) for ii in yesnoall]
	
	locall=asciiread('locall_cyk_new.txt')
	magall=asciiread('magall_cyk_new.txt')

	from aefa import acc_occurence
	import os
	aefapath=os.getenv('HOME')+"/DATALIB/AEFA.h5"
	pre,rec,acc,f1=acc_occurence(yesnoall,aefapath)
	
	'''

	#one way to get label
# 	import h5py
# 	aetalab=[]
# 	f = h5py.File(aefapath, 'r')
# 	for WKNO in range(1,31,1): 
# 		idx='WK_%02d'%(WKNO)
# 		g=f.get(idx)
# 	
# 		if g.get('Label_EV').attrs['yesno'] == 'yes':
# 			aetalab.append(1)
# 		else:
# 			aetalab.append(0)

	#an easier way to get label
	from aefa import get_testlabel
	aetalab=get_testlabel(aefapath,mode='occurence')
	
	print('aetalab size is',len(aetalab))
	print('yesno size is',len(yesno))
	
	from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

	accuracy=accuracy_score(aetalab,yesno)
	f1=f1_score(aetalab,yesno)
	precision=precision_score(aetalab,yesno)
	recall=recall_score(aetalab,yesno)

	print("Precision=",precision)
	print("Recall=",recall)
	print("Accuracy=",accuracy)
	print("F1-score=",f1)
	
	return precision,recall,accuracy,f1
	
	
def acc_mag(mags,aefapath='./'):
	'''
	acc_mag: calculate magnitude accuracy
	
	EXAMPLE
	
	from pylib.io import asciiread
	import numpy as np
	
	yesnoall=asciiread('yesnoall_cyk_new.txt');
	yesnoall=[float(ii) for ii in yesnoall]
	
	locall=asciiread('locall_cyk_new.txt')
	magall=asciiread('magall_cyk_new.txt')
	magall=[float(ii.split(" ")[0].split("]")[0].split("[")[-1]) for ii in magall] #choose the first one

	from aefa import acc_mag
	import os
	aefapath=os.getenv('HOME')+"/DATALIB/AEFA.h5"
	mae=acc_mag(magall,aefapath)
	
	'''
	import numpy as np
	from aefa import get_testlabel
	aetalab=get_testlabel(aefapath,mode='occurence')
	aetalabmag=get_testlabel(aefapath,mode='magnitude')
	
	print('aetalab size is',len(aetalab))
	
	ssum=[]
	for ii in range(len(aetalab)):
		if mags[ii]>0 and aetalab[ii]==1:
			print(mags[ii],aetalabmag[ii])
			ssum.append(np.abs(mags[ii]-aetalabmag[ii]))
	MAE=np.mean(ssum)
	
	print('MAE of magnitude=',MAE)

	return MAE
	

def acc_loc(locs,aefapath='./'):
	'''
	acc_loc: calculate location accuracy
	
	EXAMPLE
	
	from pylib.io import asciiread
	import numpy as np
	
	yesnoall=asciiread('yesnoall_cyk_new.txt');
	yesnoall=[float(ii) for ii in yesnoall]
	
	locall=asciiread('locall_cyk_new.txt')
	locall=[ii.split("]")[0].split("[")[-1] for ii in locall]
	
	magall=asciiread('magall_cyk_new.txt')
	magall=[float(ii.split(" ")[0].split("]")[0].split("[")[-1]) for ii in magall] #choose the first one

	from aefa import acc_loc
	import os
	aefapath=os.getenv('HOME')+"/DATALIB/AEFA.h5"
	mde=acc_loc(locall,aefapath)
	
	'''
	import numpy as np
	from aefa import get_testlabel
	aetalab=get_testlabel(aefapath,mode='occurence')
	aetalabmag=get_testlabel(aefapath,mode='magnitude')
	aetalabloc=get_testlabel(aefapath,mode='location')
	
	print('aetalab size is',len(aetalab))
	
	ssum=[]
	for ii in range(len(aetalab)):
		if locs[ii]!='0' and aetalab[ii]==1:
			print(locs[ii],aetalabloc[ii])
			loc=locs[ii].split()
			loc=[float(ii) for ii in loc]
			ssum.append(np.hypot(loc[0]-aetalabloc[ii][0],loc[1]-aetalabloc[ii][1])*111.0) 
	MDE=np.mean(ssum)
	
	print('MDE of location=',MDE)

	return MDE
	

