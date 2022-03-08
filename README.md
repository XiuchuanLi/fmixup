# Decision-based Adversarial Attack with Frequency Mixup

## Content

### fattack.py
Implement the *f-attack* with a limited query budget.

### vanilla_hsja.py
Implement HSJA with a relatively adequate query budget.

### revised_hsja.py
Implement HSJA equipped with *frequency binary search* with a relatively adequate query budget.

### utils

#### logger.py
Logger utilities. It improves levels of logger and add coloration for each level

#### clean_resnet.csv
Indexes of examples that are classified corretly by ResNet50.

#### clean_mobilenet.csv
Indexes of examples that are classified correctly by MobileNetv2

## Requirements
- python 3.7.4
- numpy 1.17.2
- pytorch 1.6.0
- torchvision 0.7.0

## Notes
The latest code gives better results that that in the origianl paper. Specifically, *f-attack* achieves a higher success rate and a remarkably stronger robustness against detection.
