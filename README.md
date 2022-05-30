# Decision-based Adversarial Attack with Frequency Mixup

This repository is the official implementation of [Decision-based Adversarial Attack with Frequency Mixup].

## Usage
Download and unzip the ILSVRC2012 validation set to `data/imagenet-val` folder.

(1) *f-attack*

```(bash)
python fattack.py\
    -v data/imagenet-val\
    -r {r_h}\ 
    -t {distortion threshold}\ 
    -q {query budget}\
    -m {target model: resnet or mobilenet}
```
To equip the target model with Blacklight or boundary detection, use `--bl` or `--bd`.

Compared with the vanilla *f-attack*, *dynamic f-attack* requires fewer reference examples and achieves slightly better performance.

(2) *frequency binary search*

Run `vanilla_hsja.py` to perform vanilla HSJA and `revised_hsja.py` to perform HSJA equipped with frequency binary search.

## Requirements
- python 3.7.4
- numpy 1.17.2
- pytorch 1.6.0
- torchvision 0.7.0

## Note
We have modified some details of *f-attack* and obtained better performance than that reported in the original paper.

## Reference
For technical details and full experimental results, please check the paper. If you have used our work in your own, please consider citing:

```bibtex
@article{li2022decision,
  title={Decision-Based Adversarial Attack With Frequency Mixup},
  author={Li, Xiu-Chuan and Zhang, Xu-Yao and Yin, Fei and Liu, Cheng-Lin},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={17},
  pages={1038--1052},
  year={2022},
  publisher={IEEE}
}
```
