# Quantitative analysis of patch-based fully convolutional neural networks for tissue segmentation on brain magnetic resonance imaging

This repository implements the evaluation framework proposed in one of our research paper (under evaluation). An electronic preprint is available from Arxiv:

```
Bernal, J., Kushibar, K., Cabezas, M., Valverde, S., Oliver, A., Lladó, X. (2017). "Quantitative Analysis of Patch-Based Fully Convolutional Neural Networks for Tissue Segmentation on Brain Magnetic Resonance Imaging." IEEE Access 7 (2019): 89986-90002.
```

## Requirements
### Libraries
The code has been tested with the following configuration

- h5py == 2.7.0
- ipython == 5.3.0
- jupyter == 1.0.0
- keras == 2.0.2
- nibabel == 2.1.0
- nipype == 0.12.1
- python == 2.7.12
- scipy == 0.19.0
- sckit-image == 0.13.0
- sckit-learn == 0.18.1
- tensorflow == 1.0.1
- tensorflow-gpu == 1.0.1

## How to run it
There are two main steps to run our framework. First, update parameters inside the configuration.py file. Make sure you update dataset_info inside general_configuration to your specific setup. Also, update fields on training_configuration to desired values. Approaches that can be tried are 'DolzMulti', 'Kamnitsas', 'Guerrero' and 'Cicek'. Second, run the following on command line

```
python main.py
```
