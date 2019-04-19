**Art works lab**

The goal of that lab is to create CNN that can recognize artwork author with 90% accuracy

Pre-requisites: 
1. Prepare TF environment

Steps:
1. Clone this repository: 
    `git clone git@github.com:AlexanderSoroka/CNN-ArtWorks.git`
2. Copy artworks images
3. Generate TFRecords with build_image_data.py script:
    `python build_image_data.py --data_directory=data/images --output_directory=data`
4. Run train.py to be sure the simple example is being trained
5. Modify model and have fun