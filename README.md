[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/AlexanderSoroka/CNN-ArtWorks/master/LICENSE.md)

**Art works lab**

The goal of that lab is to create CNN that can recognize artwork author with >90% accuracy

Pre-requisites: 
1. TensorFlow 1.13 environment

Steps to reproduce results:
1. Clone this repository: 
    `git clone git@github.com:AlexanderSoroka/CNN-ArtWorks.git`
2. Download [Kaggle art works dataset](https://www.kaggle.com/ikarus777/best-artworks-of-all-time) and unpack it
3. Generate TFRecords with build_image_data.py script:
    `python build_image_data.py --data_directory=data/images --output_directory=data`
    
   By default it uses 90% images for training and 10% images for validation.
4. Run train.py to train pre-defined VGG-like network. In my series of experiments that network showed 100% categorical 
accuracy on the validation split.
    
   Validation loss graph:
   
   ![](https://github.com/AlexanderSoroka/CNN-ArtWorks/blob/master/epoch_val_loss.svg)
   
   Validation categorical accuracy:
   
   ![](https://github.com/AlexanderSoroka/CNN-ArtWorks/blob/master/epoch_val_categorical_accuracy.svg)
       
5. Modify model and have fun

### [License](https://raw.githubusercontent.com/AlexanderSoroka/CNN-ArtWorks/master/LICENSE.md)

Copyright (C) 2019 Alexander Soroka.

All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only. For commercial use, please contact [soroka.a.m@gmail.com](soroka.a.m@gmail.com).
