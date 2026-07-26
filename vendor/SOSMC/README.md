# SOSMC
Repository for the "Efficient Stochastic Optimisation via Sequential Monte Carlo" paper, as found [here](https://arxiv.org/abs/2601.22003).

*Note: Each experiment comes with its own `requirements.txt` file.*

## Experiments: Reward tuning of Langevin Process
To run experimental trials for the reward tuning of Langevin processes, which note can reasonably be run on a modern personal computer, please see:
```
reward_tuning/langevin_processes/experiments.ipynb
```
For full experimental details, please refer to Appendix E.1. of the paper.

## Experiments: Reward tuning of EBMs - 2D Datasets
To run experimental trials for the reward tuning of EBMs, for the 2D datasets, please see:
```
reward_tuning/ebms_2D/experiments.ipynb
```
Also note that a collection of pre-trained models, trained using the procedure described in Appendix E.2.1. of the paper have been provided.

This code requires a modern GPU to run, for which we utilised Google Colab on a free subscription.

For full experimental details, please refer to Appendix E.2. of the paper.

## Experiments: Reward tuning of EBMs - MNIST
To run experimental trials for the reward tuning of EBMs, for MNIST, please see:
```
reward_tuning/ebms_mnist/experiments.ipynb
```
Please note that this code requires a modern GPU, for which we utilised Google Colab on a free subscription.

For full experimental details, please refer to Appendix E.3. of the paper.

## Experiments: MMLE - Image Deblurring
To run experimental trials for the image deblurring task, please see:
```
mmle/image_deblurring/setup_<setup_identifier>.ipynb
```
Please note that this code requires a modern GPU, for which we utilised Google Colab on a free subscription.

For full experimental details, please refer to Appendix E.4. of the paper.


## Citation
```
@inproceedings{
cuin2026efficient,
title={Efficient Stochastic Optimisation via Sequential Monte Carlo},
author={James Cuin and Yanbo Tang and Davide Carbone and O. Deniz Akyildiz},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
}
```