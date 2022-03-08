# Decision-based Adversarial Attack with Frequency Mixup

## Usage
Download the ILSVRC2012 validation set to "data/imagenet-val"

*f-attack*

```(bash)
python fattack.py\
    -v 'data/imagenet-val' \
    -r 0.75 {r_h}\ 
    -t 6 {distortion threshold}\ 
    -q 500 {query budget}\
    -m resnet {target model: resnet or mobilenet}\
    --bd {equip the target model with boundary detection}\
    --bl {equip the target model with blacklight}\
```


## Requirements
- python 3.7.4
- numpy 1.17.2
- pytorch 1.6.0
- torchvision 0.7.0

## Notes
The latest code gives better results than that in the origianl paper. Specifically, *f-attack* achieves a higher success rate and a remarkably stronger robustness against detection.
