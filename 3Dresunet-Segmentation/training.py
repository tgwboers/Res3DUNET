import subprocess
output = subprocess.check_output(["git", "pull"])

import os
import tables
import configparser
import pandas as pd

import matplotlib
matplotlib.use('Agg')

from model_py.UnetBuilder import UnetBuilder
from model_py.TrainManager import TrainManager
from data_py.DataLoader import DataLoader
from data_py.Hdf5Writer import Hdf5Writer
from data_py.BatchGenerator import BatchGenerator
from utils_py.visualize_utils import show_loaded_slices, show_hdf5_slices, show_train_slices, show_test_slices
from utils_py.write_utils import create_output_folders, write_predictions_to_sitk, write_slices_to_sitk

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('config.ini')
crun = config['CRUN']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    create_output_folders(config)

    if(crun.getboolean('create_hdf5')):
        # Load the training data and write to hdf5 file

        data_loader = DataLoader(config)
        data_loader.load_train_data()

        print("Saving training data as hdf5 file")
        hdf5_writer = Hdf5Writer(config, type='train')
        hdf5_writer.set_dataloader(data_loader)
        hdf5_writer.write_hdf5_file()

        print("\n---------\n")

        if(crun.getboolean('visualize')):
            print("Showing some loaded slices to check if volumes are loaded ok...")
            show_loaded_slices(config.getint('DATA', 'n_labels'), config.get('PATH', 'figures_dir')
                               + "/img_loaded.png", data_loader)


    """
    Classes that are used in this code:
    DataLoader: Loads data volumes in ITK format, and pre-processes data (resampling, cropping, normalization).
    HdfWriter: Writes data volume numpy arrays to file in hdf5 format.
    UnetBuilder: Builds a UNet Keras model with build_model method.
    TrainManager: Manages training, defines callbacks for Keras model.fit_generator.
    TestManager: Manages testing.
    """

    unet_builder = UnetBuilder(config)
    train_manager = TrainManager()  # Defines callbacks for Keras model.fit_generator


    # Build a new Unet model
    print("Building new model...")
    model = unet_builder.build_model()

    model.summary()

    df = pd.DataFrame(columns=['KFOLD', 'CaseNumber', 'Dice'])

    # Save initial model weights to use for reinitialisation later
    model_initial_weights = model.get_weights()

    """
    Train a Keras model with model.fit_generator(params) where params are:
    :param model: Keras model that will be trained. 
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return: 
    """
    # Open an hdf5 file with the training data
    hdf5_file_opened = tables.open_file(config.get('PATH','hdf5_train'), "r")  # Open hdf5 file in read mode


    batch_generator = BatchGenerator(hdf5_file=hdf5_file_opened, config=config)

    kfold = config.getint('TRAINING', 'cross_validation_kfold')
    kfold_lists = batch_generator.get_kfold_split(
        kfold_ids_file=config.get('PATH','kfold_ids_file'),
        shuffle_list=True,
        kfold=kfold)

    if (crun.getboolean('cross_validation')):
        jfold = kfold

    else:
        jfold = 0

    for ifold in range(jfold + 1):
        new_model_file = os.path.join(config.get('PATH', 'newmodel_dir'), "model_ifold_{}.h5".format(ifold))

        if (jfold == 0):
            # Single validation on first fold
            training_list = [item for jfold in range(kfold) for item in kfold_lists[jfold] if jfold != ifold]
            validation_list = kfold_lists[ifold]
            print('SINGLE VALIDATION ' + str(ifold) + '\n')
            print('TRAINING FILE LIST: ' + str(training_list) + '\n')
            print('VALIDATION FILE LIST: ' + str(validation_list) + '\n')
        elif (ifold < jfold):
            # k-fold cross-validation
            training_list = [item for jfold in range(kfold) for item in kfold_lists[jfold] if jfold != ifold]
            validation_list = kfold_lists[ifold]
            print('K-FOLD CROSS-VALIDATION, FOLD ' + str(ifold) + '\n')
            print('TRAINING FILE LIST: ' + str(training_list) + '\n')
            print('VALIDATION FILE LIST: ' + str(validation_list) + '\n')
        else:
            # Final training on all volumes after k-fold cross-validation
            training_list = [item for jfold in range(kfold) for item in kfold_lists[jfold]]
            validation_list = []
            print('FINAL TRAINING ON ALL VOLUMES, AFTER K-FOLD CROSS-VALIDATION \n')
            print('TRAINING FILE LIST: ' + str(training_list) + '\n')
            print('VALIDATION FILE LIST: ' + str(validation_list) + '\n')

        batch_size = config.getint('TRAINING', 'batch_size')
        print(training_list)
        print(validation_list)
        
        nb_train_samples = len(training_list) // batch_size
        nb_validation_samples = len(validation_list) // batch_size
        

        training_batch_generator = batch_generator.get_data_generator(index_list=training_list,
                                                                      batch_size=batch_size)
        # Python generator to provide batches of validation data as input to model.fit_generator
        validation_batch_generator = batch_generator.get_data_generator(index_list=validation_list,
                                                                        batch_size=batch_size)

        if(crun.getboolean('train_model')):

            '''
            Train the model, output a trained model config['model']['newmodel_file'] and scores for Tensorboard: 
            Run command 'tensorboard --config['data']['logging_dir']' and open webpage 'localhost:6006'.
            '''

            print("Training model...")
            # For info about inputs and outputs of fit_generator, see Keras model API at https://keras.io/models/model/
            if ifold == kfold:
                validation_data = None
            else:
                validation_data = validation_batch_generator




            print("Getting training and validation batch generators...")

            print("Training model...")
            model.set_weights(model_initial_weights)



            if(crun.getboolean('train_model')):
                # Reinitialize model weights each training fold

                model.fit_generator(generator=training_batch_generator,
                                    steps_per_epoch=nb_train_samples,
                                    epochs=config.getint('TRAINING', 'n_epochs'),
                                    validation_data=validation_batch_generator,
                                    validation_steps=nb_validation_samples,
                                    #use_multiprocessing=False,
                                    shuffle=False,
                                    verbose=1,
                                    callbacks=train_manager.get_callbacks(
                                        new_model_file,
                                        initial_learning_rate=config.getfloat('TRAINING', 'initial_learning_rate'),
                                        learning_rate_drop=config.getfloat('TRAINING', 'learning_rate_drop'),
                                        learning_rate_epochs=config.getfloat('TRAINING','decay_learning_rate_every_x_epochs'),
                                        logging_dir=config.get('PATH','logging_dir'),
                                        ifold=ifold))

                for val_case in validation_list:
                    loss, acc = model.evaluate(x=hdf5_file_opened.root.img[val_case],
                                                y=hdf5_file_opened.root.gt[val_case],
                                                batch_size=config.getint('TRAINING', 'batch_size'),
                                                verbose=0)
                    print('casenumber: {} with score: {}'.format(val_case, acc))
                    df.append({'Kfold':ifold, 'Casenumber': val_case, 'DICE':acc})


                df.to_csv(config.get('PATH', 'training_log'))

    hdf5_file_opened.close()



if(__name__ == "__main__"):
    main()

