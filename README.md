Constrained Concealment Attacks on Reconstruction-based Anomaly Detectors in Industrial Control Systems
=======
 
## Implementation of iterative and learning-based concealment attacks
  
### Description
  
 This repository is organized as follows:

  * `Attacked_Model` contains the Autoencoder based detector trained on BATADAL and WADI data. The code was forked from [AutoEncoders for Event Detection (AEED)](https://github.com/rtaormina/aeed) and complemented with LST and CNN defenses. Please refer to the original repository to train your defense model.

  * `Adversarial_Attacks` contains the iterative and learning-based concealment attacks implementations used for the evaluation of our manuscript.

  * `Data` contains the dataset used for the experiment. *Note* to obtain WADI data please refer to [iTrust](https://itrust.sutd.edu.sg/)

  * `Evaluation` contains the python notebooks to evaluate the attack efficacy in different constraints scenarios. By running each notebook you reporduce the results found in the paper and the relative plots (i.e. Figures 2 and 3 in the manuscript). To open jupyter run the following commad `conda activate && jupyter notebook` and then open the .ipynb files in Evauation folder

### Requirements

To execute attacks and evaluate them `Python 3`, `Keras`, `Pandas`, `Numpy`,  `Jupyter` are required. Installation through `conda` is suggested.

### Usage
Command-line options available: 

`-d 'dataset'`data to be used for the attacks, default=BATADAL

#### Whitebox_Attack folder:

##### unconstrained_attack.py

It performs the unconstrained iterative attack in the white box setting. 

Usage:

`python unconstrained_attack.py -d BATADAL`

##### constrained_attack_constraints_extraction.py

Extract the constraints for the best-case scenario constrained attack. 

Usage:

`python constrained_attack_constraints_extraction.py -d BATADAL`

##### constrained_attack.py  

It performs the best-case scenario constrained attack, and applies the constraints found with `constrained_attack_constraints_extraction.py`.

Usage:

`python constrained_attack.py -d BATADAL`

##### constrained_attack_PLC.py  

It performs the topology-based scenario constrained attack, and applies the constraints according to the PLC controlled by the attacker. 

Optional Parameters: 

* `-a 'attack id'` attack id that we want to conceal

* `-p 'PLC_N'` PLC controlled by the attacker

Usage:

`python constrained_attack_PLC.py -d BATADAL -a 1 -p PLC_1`

##### launcher.py

util to automate and parallelize topology-based scenario constrained attacks.

Usage:

`python launcher.py -d BATADAL`


#### Black_Box_Attack folder:
Optional Parameter:

* `-p 'Bool'` train Adversarial Autoencoder network,  default=False (All trained models are already available in the repository)

##### unconstrained_attack.py

It performs the unconstrained learning-based attack in the black box setting. 

Optional Parameters:

* `-t 'Bool'`, measure time for concealment attack,  default=False

Usage:


`python unconstrained_attack.py -d BATADAL`


##### constrained_attack_X_dimension.py

Partially constrained attack best-case constraints

Optional Parameters:

* `-f 'Bool'`, default=True. If `False` extract the constraints for the best-case scenario constrained attack. (All saved constraints are available in the repository)

Usage:

`python constrained_attack_X_dimension.py -d BATADAL -p False -f True`

##### constrained_attack_X_dimension_PLC.py

Partially constrained attack topology-based constraints script, X Dimension, PLC constraints 

Usage: 

`python constrained_attack_X_dimension_PLC.py -d BATADAL -p False`


##### constrained_AE_attack_X_dimension_PLC.py        
Fully constrained attack script, topology-based constraints, X Dimension, PLC constraints 

Usage:

`python constrained_AE_attack_X_dimension_PLC.py -d BATADAL -p False`



##### constrained_attack_D_dimension.py               

Constrained attack, D dimension

Usage:

`python constrained_attack_D_dimension.py -d BATADAL -p False`


#### Replay_Attack folder:

##### replay_attack.py

Optional Parameters:

* `-c 'topology'/'best'` constraints to be applied scenario topology or best case scenario, default=best

Usage:

`python replay_attack.py -d BATADAL -c best`

