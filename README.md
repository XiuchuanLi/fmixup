# Decision-based Adversarial Attack with Frequency Mixup
It has been widely observed that deep neural networks are highly vulnerable to adversarial examples. Decision-based attacks could generate adversarial examples based solely on top-1 labels returned by the target model. However, they typically make excessive queries and could not bypass detection effectively. To comprehensively assess a decision-based attack, besides its query efficiency, the performance against detection is also a concern. Considering that previous detections consume massive resources and always mistakenly recognize benign video frames as malicious attacks, we design a lightweight detection called *boundary detection* to overcome the above limitations, whose success reveals serious limitations of existing decision-based attacks. To develop more powerful attacks, we first present *f-mixup* as a basic method to produce candidate adversarial examples in the frequency domain. Using *f-mixup* as the building block, we propose *f-attack* as a complete decision-based attack. With the help of several natural images, *f-attack* could both work well with limited (hundreds of) queries and bypass detection effectively. Nevertheless, if the attacker could make relatively adequate (thousands of) queries and the target model is not equipped with detection, *f-attack* will lag behind existing decision-based attacks. We additionally introduce *frequency binary search* based on *f-mixup*, which serves as a plug-and-play module for existing decision-based attacks to further improve their query efficiency. Experimental results verify the effectiveness of our proposed methods.

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
