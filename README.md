# GeoFWI

## Description

**GeoFWI** is a lightweight velocity model dataset for deep-learning-based FWI benchmarking. This project is inspired by the OpenFWI project (https://sites.google.com/site/youzuolin044/openfwi) but includes more realistic geological models (e.g., folding layers, faults, and salt bodies). 

Currently, we call it lightweight because this is the first step toward an eventually "heavyweight," very realistic, highly generalizable 3D velocity model dataset for unprecedented high-efficacy deep-learning-empowered FWI studies. 

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
## Development
    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 

-----------
## Contact
    Regarding any questions, bugs, developments, collaborations, please contact  
    Yangkang Chen
    chenyk2016@gmail.com

-----------
## License
    MIT License
    
-----------
## Download GeoFWI.npy from
https://utexas.box.com/s/scbh25utyy5jz3mq7b0hp3wyluka7gaq

## Reference
Li et al., 2025, GeoFWI - upcoming, TBD. 

Li, S., Liu, B., Ren, Y., Chen, Y., Yang, S., Wang, Y. and Jiang, P., 2020. Deep-learning inversion of seismic data, IEEE Transactions on Geoscience and Remote Sensing, 58, 3, 2135-2149.

Liu, B., Yang, S., Ren, Y., Xu, X., Jiang, P. and Chen, Y., 2021. Deep-learning seismic full-waveform inversion for realistic structural models. Geophysics, 86(1), pp.R31-R44.

Ren, Y., Nie, L., Yang, S., Jiang, P. and Chen, Y., 2021. Building complex seismic velocity models for deep learning inversion. IEEE Access, 9, pp.63767-63778.

BibTeX:

	@Article{seisinvnet,
  	author={Shucai Li and Bin Liu and Yuxiao Ren and Yangkang Chen and Senlin Yang and Yunhai Wang and Peng Jiang},
  	title = {Deep learning Inversion of Seismic Data},
  	journal={IEEE Transactions on Geoscience and Remote Sensing},
  	year=2020,
  	volume=58,
  	number=3,
  	issue=3,
  	pages={2135-2149},
  	doi={10.1109/TGRS.2019.2953473},
	}

	@Article{seisinvnet2,
  	author={Bin Liu and Senlin Yang and Yuxiao Ren and Xinji Xu and Peng Jiang and Yangkang Chen},
  	title = {Deep learning seismic full waveform inversion for realistic structure models},
  	journal={Geophysics},
  	year=2021,
  	volume=86,
  	issue=1,
  	number=1,
  	pages={R31-R44},
  	doi={10.1190/geo2019-0435.1},
	}

	@Article{velocitymodel,
  	author={Yuxiao Ren and Lichao Nie and Senlin Yang and Peng Jiang and Yangkang Chen},
  	title = {Building Complex Seismic Velocity Models for Deep Learning Inversion},
  	journal={IEEE Access},
  	year=2021,
  	volume=9,
  	issue=1,
  	number=1,
  	pages={63767-63778},
  	doi={10.1109/ACCESS.2021.3051159},
	}

