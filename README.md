# FlowCamClassification

Code associated with the paper `Collaborative deep learning models to handle class imbalance in FlowCam plankton imagery`. Although functionality could be imported into another project, this was designed to be a command line tool that includes all functionality from processing FlowCam images to data clean up, training of deep learning models all to reporting the performance of different types of models.

## How to cite
T. Kerr, J. R. Clark, E. S. Fileman, C. E. Widdicombe and N. Pugeault, "Collaborative deep learning models to handle class imbalance in FlowCam plankton imagery," in IEEE Access, doi: 10.1109/ACCESS.2020.3022242.


## Prerequisites 

It's recommended to create a clean Python (version >= 3.7) environment using Anaconda or similar, for example:

``` conda create -n yourenvname python=3.7 anaconda ```


With the environment activated (for example `conda activate yourenvname`), install the projects dependencies with:

  

``` pip install -r requirements.txt ```

  

## Getting started
Please create empty `processed_data` and `reports` directories in the root directory.

Typical workflow would start with processing the raw flowcam data into a machine
learning friendly format (training, testing and validation sets for both the images
and the geometrics). From there various individual deep learning models can be 
trained of different scales (VGGNet, GoogLeNet, ResNet & DenseNet) while the geometric data can be trained using an MLP model. If desired multple pretrained models can be loaded into a new collaborative model which learns how each of these learners should collaborate. Finally, there is some additionaly functionality to create various reports.

Note: it's usually easier to manually configure the default constants near the top of `main.py` rather than passing them in as command line arguments. Then for example just using `python main.py train_cnn`

### Creating training data from FlowCam files

Raw FlowCam data should go in the `flowcam_data` directory, with a sub-directory for every type of particle. We included some data as an example. Each sub directory should contain at least one image collage. For each image collage there should contain a corresponding `flb` or `lst` file. Finally to generate a machine learning friendly dataset from these files, run the following command, where min_samples will filter out any taxonomic groups where the number of particles fall below the given value:

  

``` python main.py create_training_data --flowcam_data_dir REPLACE_IF_FILES_NOT_IN_FLOWCAM_DATA --min_samples MINIMUM_NUMBER_OF_PARTICLES_TO_BE_CONSIDERED --desired_image_size```

  

This will generate some new files in `processed_data` directory:

`class_dict.npy` - A dictionary mapping ids that a machine learning model will use to human readable labels.

`original_data.csv` - A master CSV file containing all the original data scraped from ALL the `flb` or `lst` files, before any operations were performed on the data.

`plankton_data_101x64_final.pkl` - The final dataset split into training, validation and testing sets. This can be imported using the `pickle` module. This file contains:

Geometrics : `trainAttrX, valAttrX, testAttrX`

Images: `trainImagesX, valImagesX, testImagesX`

Labels: `y_train, y_val, y_test`

`std_scaler.bin` - Geometric data is standardized, this allows for the data to be transformed back to it's original values.

  

### Train a Convolutional Neural Network (ConvNet) model

This project has 4 types of ConvNet modules implemented, inspired from: GoogLeNet(with or without residual connections), ResNet, VGGNet and DenseNet. Each module implements the main innovation in each model type. These should be scaled up by passing the `num_modules` parameter.

(worth noting, default parameters are set if none are passed)

  

``` python main.py train_cnn --convnet_model_type ONE_OF[InceptionResidual, Inception, Residual, VGG, Dense] --num_modules --cnn_model_name --num_epochs --batch_size```

  

A snapshop of the top performing model during training (using validation accuracy) is saved in `trained_models` directory.

  

### Train a Multi-layer Perceptron (MLP) model

``` python main.py train_mlp --mlp_model_name ```

### Train a Collaborative model

#### WARNING (TODO) - currently hardcoded to only accept two ConvNets and MLP (three models in total), `train_multi_model_collab` and `create_augmented_images_generator` need to be adjusted to accept more/less models.
By passing the filenames of one or more trained ConvNets and an MLP model, this loads each model, freezes the weights and then creates a new collaborative model for training. This model is then compiled and carries out a training phase via TensorFlow to learn the collaborative function within the new model. The best model eventually gets saved in the `trained_models` directory by default. Please edit `DEFAULT_BEST_MODELS` in `main.py` to configure a list of ConvNet models (these should all be stored in `trained_models` by default) to use, the MLP model name can be passed through a command line argument. 

``` python main.py train_multi_cnn_colab_model --mlp_model_name ```