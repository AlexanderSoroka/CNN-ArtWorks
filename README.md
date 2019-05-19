**Art works lab**

The goal of that lab is to create CNN that can recognize artwork author with >90% accuracy

Pre-requisites: 
1. TensorFlow 1.13 environment

Steps to reproduce results:
1. Clone this repository: 
    `git clone git@github.com:AlexanderSoroka/CNN-ArtWorks.git`
2. Download [Kaggle art works dataset](https://www.kaggle.com/ikarus777/best-artworks-of-all-time)
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