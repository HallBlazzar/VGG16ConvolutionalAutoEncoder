# This is VGG16 AutoEncoder developed by Keras and Tensorflow backend.

## Requirement ##

* python3.6
* keras
* Tensorflow
* cv2
* h5

## How to use ##

Under project directory:

* `data_source`: 

    Put training data and testing data there(separated by sub-directory).

* `model_and_weight`: 

    Basically, `vgg16_weights_notop.h5` is necessary.

    And generated models and weight will be stored there.

* `ModelTrainer.py`: 

    Run this to train model.(trained through `data_source\training` and evaluate through `data_source\testing`)

* `AutoEncoderEfficiencyEvaluator.py`

    Run this to generate images after encoded and decoded.(data within `data_source\testing`)

    The images would be stored in `auto_encoder` sub-directory.
