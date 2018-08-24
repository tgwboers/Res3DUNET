import os
import sys
from time import time


import numpy as np
import tables
import configparser

import matplotlib
matplotlib.use('Agg')

import csv
from keras.utils import to_categorical
from keras.models import load_model

from model_py.UnetBuilder import UnetBuilder
from model_py.TrainManager import TrainManager
from model_py.TestManager import TestManager
from data_py.DataLoader import DataLoader
from data_py.Hdf5Writer import Hdf5Writer
from data_py.BatchGenerator import BatchGenerator
from utils_py.visualize_utils import show_loaded_slices, show_hdf5_slices, show_train_slices, show_test_slices
from utils_py.write_utils import create_output_folders, write_predictions_to_sitk, write_slices_to_sitk

config = configparser.ConfigParser()
config.read('config.ini')
crun = config['CRUN']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    print("Loading test data from ", config.get('PATH', 'test_dir'), "...")
    data_loader_test = DataLoader(config=config)
    data_loader_test.load_test_data(with_gt=True)

    print("Saving test data as hdf5 file")
    hdf5_writer_test = Hdf5Writer(config, type='test')
    hdf5_writer_test.set_dataloader(data_loader_test)
    hdf5_writer_test.write_hdf5_file()

    test_manager = TestManager()

    # the dice scores as defined in Unet_builder.
    print("Loading test data from ", config.get('PATH', 'hdf5_test'), " ...")
    with_gt = True
    data_loader = DataLoader(config)  # empty data_loader
    data_loader.load_test_data(with_gt=with_gt)


    """
    Classes that are used in this code:
    DataLoader: Loads data volumes in ITK format, and pre-processes data (resampling, cropping, normalization).
    HdfWriter: Writes data volume numpy arrays to file in hdf5 format.
    UnetBuilder: Builds a UNet Keras model with build_model method.
    TrainManager: Manages training, defines callbacks for Keras model.fit_generator.
    TestManager: Manages testing.
    """

    unet_builder = UnetBuilder(config)

    # Build a new Unet model
    print("Building new model...")
    model = unet_builder.build_model()

    #weights = load_weights(os.path.join(config.get('PATH', 'newmodel_dir'), "new_model.h5"))
    #model.set_weights(weights)

    print("Testing model...")
    test_pred, test_gt, test_dice = test_manager.test_model(data_loader, model, unet_builder.dice_coef,
                                                            config.getint('DATA', 'n_labels'), with_gt)
    print(" ")
    n_labels = config.getint('DATA', 'n_labels')
    print('n_labels: ',n_labels)
    print("{} {}".format('Patient', '[Dice scores per label]'))
    with open('test_dice_scores.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["n_labels: ", n_labels])
        csv_writer.writerow(config['data']['label_names'][:n_labels])
        csv_writer.writerow(config['data']['label_reassign'][:n_labels+1])
        for pat, dice in test_dice.items():
            print(pat, dice)
            #csv_writer.writerow([pat]+str(dice))

    visualize=False
    if(visualize):
        print("Showing some predicted test slices...")
        show_test_slices(config['data']['n_labels'], config['data']['figures_dir']+"/img_pred.png",
                         with_gt, data_loader, test_pred, 0.5)

    write_sitk = True
    if(write_sitk):
        print("Writing predicted test volumes to ITK...")
        # Write predictions to .mhd/.raw files that can be inspected in ITK-SNAP,
        # both in the original resolution and in the Unet input resolution.
        write_predictions_to_sitk(config['data'], data_loader, test_pred)

    print("extra line to set break point for debugging")

if(__name__ == "__main__"):
    main()
