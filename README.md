# GeoFWI

## Description

**GeoFWI** is a lightweight velocity model dataset for deep-learning-based FWI benchmarking. This project is inspired by the OpenFWI project (https://sites.google.com/site/youzuolin044/openfwi) but includes more realistic geological models (e.g., folding layers, faults, and salt bodies). 

Currently, we call it lightweight because this is the first step toward an eventually "heavyweight," very realistic, highly generalizable 3D velocity model dataset for unprecedented high-efficacy deep-learning-empowered FWI studies. 

-----------
## Download GeoFWI.npy from
https://utexas.box.com/s/scbh25utyy5jz3mq7b0hp3wyluka7gaq

## Reference
Li et al., 2025, GeoFWI: A lightweight velocity model dataset for benchmarking full waveform inversion using deep learning - upcoming, TBD. 

BibTeX:

	@Article{geofwi,
  	author={Chao Li and Liuqing Yang and Sergey Fomel and Andrey Bakulin and Yiran Shen and Shuang Gao and Umair Bin Waheed and Alexandros Savvaidis and Yangkang Chen},
  	title = {GeoFWI: A lightweight velocity model dataset for benchmarking full waveform inversion using deep learning},
  	journal={TBD},
  	year=2025,
  	volume=TBD,
  	number=TBD,
  	issue=TBD,
  	pages={TBD},
  	doi={TBD},
	}

-----------
## Copyright
    GEOFWI developing team, 2024-present

-----------
## License
    MIT License 

-----------
## Install
Using the latest version

    git clone https://github.com/aaspip/geofwi
    cd geofwi
    pip install -v -e .

or using Pypi

    pip install geofwi

or (recommended, because we update very fast)

	pip install git+https://github.com/aaspip/geofwi

-----------
## Verified runnable OS
Mac OS, Linux, Windows (need Microsoft C++ Build Tools)

-----------
## Examples

Here is a sneak peek of some of the structures and initial results of the GeoFWI project.

<p align="center">
<img src='https://github.com/aaspip/gallery/blob/main/geofwi/geofwi-seisinvnet.png' alt='comp' width=960/>
</p>

Figure below shows a few good/bad predictions using the data-driven DL-based method (SeisInvNet) detailed in Li et al., 2020, Deep learning Inversion of Seismic Data, IEEE TGRS.

<p align="center">
<img src='https://github.com/aaspip/gallery/blob/main/geofwi/geofwi-types.png' alt='comp' width=960/>
</p>

-----------
## Play with GeoFWI

Get some statistics from GeoFWI

	import numpy as np
	sizes=np.load("data/geofwi-size-layer-fault-salt-1-10.npy")
	print('length of sizes:',len(sizes))
	
	print('Two-layer folding model: %d samples'%sizes[0])
	print('Two-layer fault model: %d samples'%sizes[1])
	print('Two-layer salt model: %d samples'%sizes[2])
	print('Three-layer folding model: %d samples'%sizes[3])
	print('Three-layer fault model: %d samples'%sizes[4])
	print('Three-layer salt model: %d samples'%sizes[5])
	print('Four-layer folding model: %d samples'%sizes[6])
	print('Four-layer fault model: %d samples'%sizes[7])
	print('Four-layer salt model: %d samples'%sizes[8])
	print('Five-layer folding model: %d samples'%sizes[9])
	print('Five-layer fault model: %d samples'%sizes[10])
	print('Five-layer salt model: %d samples'%sizes[11])
	print('Six-layer folding model: %d samples'%sizes[12])
	print('Six-layer fault model: %d samples'%sizes[13])
	print('Six-layer salt model: %d samples'%sizes[14])
	print('Seven-layer folding model: %d samples'%sizes[15])
	print('Seven-layer fault model: %d samples'%sizes[16])
	print('Seven-layer salt model: %d samples'%sizes[17])
	print('Eight-layer folding model: %d samples'%sizes[18])
	print('Eight-layer fault model: %d samples'%sizes[19])
	print('Eight-layer salt model: %d samples'%sizes[20])
	print('Nine-layer folding model: %d samples'%sizes[21])
	print('Nine-layer fault model: %d samples'%sizes[22])
	print('Nine-layer salt model: %d samples'%sizes[23])
	print('Ten-layer folding model: %d samples'%sizes[24])
	print('Ten-layer fault model: %d samples'%sizes[25])
	print('Ten-layer salt model: %d samples'%sizes[26])
	print('Eleven-layer folding model: %d samples'%sizes[27])
	print('Eleven-layer fault model: %d samples'%sizes[28])
	print('Eleven-layer salt model: %d samples'%sizes[29])
	
The output will be like

> Two-layer folding model: 2826 samples 

> Two-layer fault model: 2754 samples 

> Two-layer salt model: 1888 samples

> Three-layer folding model: 2649 samples

> Three-layer fault model: 2473 samples
> Three-layer salt model: 1758 samples
Four-layer folding model: 2491 samples
Four-layer fault model: 2161 samples
Four-layer salt model: 1674 samples
Five-layer folding model: 2245 samples
Five-layer fault model: 1948 samples
Five-layer salt model: 1508 samples
Six-layer folding model: 1974 samples
Six-layer fault model: 1653 samples
Six-layer salt model: 1340 samples
Seven-layer folding model: 1797 samples
Seven-layer fault model: 1478 samples
Seven-layer salt model: 1226 samples
Eight-layer folding model: 1635 samples
Eight-layer fault model: 1325 samples
Eight-layer salt model: 1076 samples
Nine-layer folding model: 1453 samples
Nine-layer fault model: 1117 samples
Nine-layer salt model: 976 samples
Ten-layer folding model: 1309 samples
Ten-layer fault model: 1000 samples
Ten-layer salt model: 881 samples
Eleven-layer folding model: 1200 samples
Eleven-layer fault model: 829 samples
Eleven-layer salt model: 832 samples

-----------
## Development
    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 

-----------
## Contact
    Regarding any questions, bugs, developments, collaborations, please contact  
    Yangkang Chen
    chenyk2016@gmail.com
    

-----------
## Gallery
The gallery figures of the geofwi project can be found at
    https://github.com/aaspip/gallery/tree/main/geofwi
All figures in the gallery directory can be reproduced by their corresponding DEMO scripts in the "demo" directory. These gallery figures are also presented below. 

# Below is an example for quickly visualizing a few samples from GeoFWI.npy in different scenarios 
Generated by [demos/test_quick_visualize.py](https://github.com/aaspip/geofwi/tree/main/demos/test_quick_visualize.py)

First scenario: randomly chosen 25 samples from the whole GeoFWI dataset
<img src='https://github.com/aaspip/gallery/blob/main/geofwi/geofwi-samples-random.png' alt='comp' width=960/>

Second scenario: the first 25 samples from the whole GeoFWI dataset
<img src='https://github.com/aaspip/gallery/blob/main/geofwi/geofwi-samples-increasing.png' alt='comp' width=960/>

Third scenario: the last 25 samples from the whole GeoFWI dataset
<img src='https://github.com/aaspip/gallery/blob/main/geofwi/geofwi-samples-increasing.png' alt='comp' width=960/>

Fourth scenario: the last 25 samples in the "11-layer fault category" (e.g., sizes=np.load('data/geofwi-size-layer-fault-salt-1-10.npy'); index is chosen from 49476-sizes[-1]-1 to sizes[-1]-25) from the whole GeoFWI dataset
<img src='https://github.com/aaspip/gallery/blob/main/geofwi/geofwi-samples-faults.png' alt='comp' width=960/>