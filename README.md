# Semantic Adversarial Toolbox

Semantic Adversarial Toolbox is a python library for generating and defending
against semantic adversarial attacks. 

## Usage Guide

To install the toolbox, clone or download the repo containing the code. 
Install the required python packages specified in the 'requirements.txt'
file. 

This toolbox is built for PyTorch and will require PyTorch models for all
attacks. 


### Running a simple attack

Attacks in SAT require 3 inputs during initialization. These are:

1. Target classifier 
2. Pytorch device to store Tensors.
3. A normalization transform function for image normalization, since this cannot be
done beforehand. 
   
The following code snippet will run the HSV attack on an imagenet sample:

```python
from sat.attacks.color import HSVAttack

hsv = HSVAttack(model, device='cpu', norm=imagenet_norm)
adversarials = hsv(images, labels, max_trials=100)
```

The attacks expects batched inputs, so the output will be a list of
adversarial examples. The following code will visualize the results.


```python
adv = adversarials[0]
adv.visualize()
```

The result of the adversarial attack is shown in the image below. 

![alt text](https://i.imgur.com/C5y7rg4.png "HSV attack visualization.")


### Adversarial Training

To use adversarial training, a few parameters are needed. The model parameter
is the target classifier of the attacks. Since multiple attacks are
supported, the toolbox expects a list of attacks to apply. The 'args'
parameter defines the different parameters of each attack. 


```python

args = {
    'HSVAttack': {
        'generate_samples': True,
        'stop_on_success': True,
        'include_unsuccessful': False,
        'max_trials': 100,
        'by_score': True
    }
}

advtrain = AdversarialTraining(model,
                               [HSVAttack],
                               args,
                               device=device,
                               norm=cifar_norm,
                               transforms_train=transform_train,
                               transforms_test=transform_test)
```


Adversarial training is split into 3 steps. The first step is
generating adversarial images from a dataset. The follwing
code snippet acomplished this task:

```python
adv_samples = advtrain.generate(trainloader,
                                early_stopping=True,
                                num_adv=25000,
                                to_file=False)
```

The second step is generating an adversarial dataset consisting of both
adversarial samples and non-adversarial samples.

```python
adv_trainloader, adv_valloader = advtrain.make_loader([adv_samples],
                                                      trainloader,
                                                      val_ratio=0.1,
                                                      fixed_dataset_size=False)
```

The third step consists of training a new classifier using the generated
adversarial dataset.


## Running demos

Several demos are included in the repo under the 'examples' folder. 
These examples should provide a decent introduction to using the toolbox. 
When running any included demo file, make sure to set the repo root as 
working directory, otherwise some file paths may be invalid. 
