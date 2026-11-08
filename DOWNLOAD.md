## Download GeoFWI.npy from
https://zenodo.org/records/24061019

or
https://utexas.box.com/s/scbh25utyy5jz3mq7b0hp3wyluka7gaq

## Download small shot-gather database for simple training tests from
https://utexas.box.com/s/t6rmre575qxb2x0yvdanst6pamzvsnxp

NOTE 1: Please put the folders train/valid/test in the geofwi_train folder, which includes 50 samples for training, 10 samples for validation, and 111 samples for the completely independent test.

NOTE 2: The test set is not used in training the best SeisInvNet model, trained from 40000 samples, validated on 5000 samples, and tested on 4476 samples. It is downloadable from 
https://utexas.box.com/s/6ak8omw3rlvatdnh51ngig1n94e02mss

NOTE 3: For larger-scale training, e.g., a scale of thousands or tens of thousands of velocity samples, we recommend generating the shot gathers by the users themselves. That's why we only focus on a "lightweight" velocity model dataset (not shot gathers) in the GeoFWI project.


