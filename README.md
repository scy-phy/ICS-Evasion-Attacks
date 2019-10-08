Real-time Evasion Attacks with Physical Constraints on Deep Learning-based Anomaly Detectors in Industrial Control Systems
=======
 
Implementation of white box and black box classifier evasion from the techincal report: [Real-time Evasion Attacks with Physical Constraints on Deep Learning-based Anomaly Detectors in Industrial Control Systems](https://arxiv.org/abs/1907.07487) Erba et al. 2019
  
### Description
  
 This repository is organized as follows:

  * `Attacked_Model` contains the Autoencoder based detector trained on BATADAL and WADI data. If you want to train you own models please refer to [AutoEncoders for Event Detection (AEED)](https://github.com/rtaormina/aeed)

  * `Adversarial_Attacks` contains the white and black box implementations as described in the [techincal report](https://arxiv.org/abs/1907.07487)

  * `Data` contains the dataset used for the experiment. *Note* to obtain WADI data please refer to [iTrust](https://itrust.sutd.edu.sg/)

  * `Evaluation` contians the script to evaluate the attack efficacy

### Requirements

In order to execute attacks and evaluate them `Python 3`, `Keras`, `Pandas`, `Numpy` are required.

## Citation
When citing this work, you should use the following bibtex:

    @article{erba2019real,
      title={Real-time Evasion Attacks with Physical Constraints on Deep Learning-based Anomaly Detectors in Industrial Control Systems},
      author={Erba, Alessandro and Taormina, Riccardo and Galelli, Stefano and Pogliani, Marcello and Carminati, Michele and Zanero, Stefano and Tippenhauer, Nils Ole},
      journal={arXiv preprint arXiv:1907.07487},
      year={2019}
    }