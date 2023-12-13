# Amalgamating Multi-Task Models with Heterogeneous Architectures (AmalMTH)
This is the implementation repository of <i>Versatile Common Feature Consolidator (VENUS)</i>, the solution for AmalMTH, AAAI 2024.

## File listing

+ __main.py__ : Code for training VENUS
+ __model.py__ : Supporting models
+ __utils.py__ : Supporting utility functions
+ __requirements.txt__ : Library requirements

## Instructions 

Prepared folders:

+ __data__ : directory to place training data
+ __teachers__ : directory to place teacher models
+ __output__ : directory for student model as outputs 

The datasets and the teachers we use in our paper are available here: 
https://drive.google.com/drive/folders/1ShzRZF2ARXTnyfEaS-uRdVehX7bt8wlJ?usp=drive_link


Run script as:

    python main.py -backbone resnet -tname t0_densenet t1_resnet -t_modelname densenet resnet -data-name data1 -t_tasks '2 0' '2 3 1' -ep 1

<b>Parameters:</b>

+ __Required:__
  + __-backbone__ : the backbone model for the student model e.g., resnet
  + __-tname__ : the name of teacher models e.g., t0_densenet, t1_resnet
  + __-t_modelname__ : the architecture of each teacher e.g., densenet, resnet
  + __-t_tasks__ : teachers’ tasks e.g., t0: '2 0', t1: '2 3 1'
  + __-dataname__ : unlabelled data for training the student

+ __Hyperparameters:__
  + __-lr__ : learning rate, default 0.01
  + __-ep__ : epochs, default 50
  + __-bs__ : batch size, default 16
