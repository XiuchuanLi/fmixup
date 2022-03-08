# Decision-based Adversarial Attack with Frequency Mixup

## Usage
Download and unzip the ILSVRC2012 validation set to "data/imagenet-val" folder.

(1) *f-attack*

```(bash)
python fattack.py\
    -v data/imagenet-val\
    -r {r_h}\ 
    -t {distortion threshold}\ 
    -q {query budget}\
    -m {target model: resnet or mobilenet}\
```
To equip the target model with Blacklight or boundary detection, use `--bl` or `--bd`.

(2) frequency binary search

Run `vanilla_hsja.py` to perform vanilla HSJA and `revised_hsja.py` to perform HSJA equipped with frequency binary search.

## Requirements
- python 3.7.4
- numpy 1.17.2
- pytorch 1.6.0
- torchvision 0.7.0

## Notes
The latest code gives better results than that in the origianl paper. Specifically, *f-attack* achieves a higher success rate and a remarkably stronger robustness against detection.
