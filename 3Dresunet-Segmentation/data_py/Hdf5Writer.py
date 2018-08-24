import os

import numpy as np
import tables

class Hdf5Writer(object):
    """
    Writes data volume numpy arrays to file in hdf5 format, for fast access during training without having
    to keep all data in memory.
    """

    def __init__(self, config, type='train'):
        """
        :param data_loader: the data loader that loads and pre-processes the data and outputs the data volumes
        as numpy arrays.
        :param config_data: data configuration parameters.
        """
        self.config = config
        self.hdf5_name = config.get('PATH', 'hdf5_' + type)
        self.hdf5_file = None

    def set_dataloader(self, data_loader):
        self.data_loader = data_loader

    def write_hdf5_file(self):
        # Create hdf5 file and fill it with data volumes.
        try:
            if os.path.exists(self.hdf5_name):
                os.remove(self.hdf5_name)
            self.hdf5_file = tables.open_file(self.hdf5_name, mode='w')
            img_storage, gt_storage = self.create_hdf5_file()
            self.fill_hdf5_file(img_storage, gt_storage)
            self.hdf5_file.close()  # Close hdf5 file in write mode
        except Exception as e:
            # If something goes wrong, delete the incomplete data file
            if os.path.exists(self.hdf5_name):
                self.hdf5_file.close()  # Close hdf5 file in write mode
                os.remove(self.hdf5_name)
            raise e

    def create_hdf5_file(self):
        filters = tables.Filters(complevel=5, complib='blosc:zlib')
        shape = np.append(0, self.data_loader.img_numpy[self.data_loader.img_list[0]].shape)
        img_storage = self.hdf5_file.create_earray(self.hdf5_file.root, 'img', tables.Float32Atom(),
                                                   shape=shape, filters=filters,
                                                   expectedrows=len(self.data_loader.img_list))
        gt_storage = self.hdf5_file.create_earray(self.hdf5_file.root, 'gt', tables.UInt8Atom(),
                                                    shape=shape, filters=filters,
                                                    expectedrows=len(self.data_loader.gt_list))


        return img_storage, gt_storage


    def fill_hdf5_file(self, img_storage, gt_storage):
        # self.hdf5_file.create_array(self.hdf5_file.root, "affine", affine)
        for img_name in self.data_loader.img_list:
            filename, ext = os.path.splitext(img_name)
            filename, ext = os.path.splitext(filename)
            # gt_name = join(filename + '_segmentation' + ext)
            gt_name = filename + '_segmentation.nii.gz'
            # print("Image name in fill_hdf5_file ", img_name)
            img_storage.append(self.data_loader.img_numpy[img_name][np.newaxis])
            # print("Ground truth name in fill_hdf5_file ", gt_name)
            gt_storage.append(self.data_loader.gt_numpy[gt_name][np.newaxis])

        return

    def close_hdf5(self):
        self.hdf5_file.close()  # Close hdf5 file in write mode