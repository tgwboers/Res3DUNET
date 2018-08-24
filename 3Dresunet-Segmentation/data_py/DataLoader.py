import os
import numpy as np
import SimpleITK as sitk


class DataLoader(object):
    """
    Loads data volumes in ITK format, pre-processes data (resampling, cropping, normalization), converts
    each data volume to a numpy array and organises data volumes in a list of numpy arrays.
    :return: The following lists are accessed by the module Hdf5Writer:
    data_loader.img_numpy: list of training volume numpy arrays
    data_loader.gt_numpy: list of ground truth volume numpy arrays
    data_loader.img_list: list of names of training volume files
    data_loader.gt_list: list of names of ground truth volume files
    """

    def __init__(self, config):
        """
        :param config: configuration parameters.
        """
        self.grid_size = eval(config.get('DATA', 'grid_size'))
        self.grid_resolution = eval(config.get('DATA', 'grid_resolution'))
        self.padding = eval(config.get('DATA','padding'))
        self.orig_shape = [None] * 3
        self.train_dir = os.path.abspath(config.get('PATH', 'train_dir'))
        self.train_seg_dir = os.path.abspath(config.get('PATH', 'train_dir'))
        self.test_dir = os.path.abspath(config.get('PATH', 'test_dir'))
        self.test_seg_dir = os.path.abspath(config.get('PATH', 'test_seg_dir'))


        self.img_list = list()
        self.gt_list = list()
        self.img_sitk = dict()
        self.img_crop = dict()
        self.gt_sitk = dict()
        self.img_numpy = None
        self.gt_numpy = None

        self.new_size = None
        self.mean_intensity_train = None

    def load_train_data(self):
        print("Loading training data from ...", self.train_dir)
        print("Grid resolution: ", self.grid_resolution)
        print("Grid size: ", self.grid_size)

        self.create_images_list(self.train_dir)
        self.load_images(self.train_dir)
        self.img_numpy = self.get_data_numpy(self.img_sitk, self.img_list, resample=False, crop=False, normalize=False)

        self.create_ground_truths_list()
        self.load_ground_truths(self.train_dir)
        self.gt_numpy = self.get_data_numpy(self.gt_sitk, self.gt_list, resample=False, crop=False, normalize=False)
        self.gt_numpy = self.discrete_gt_label(self.gt_numpy, self.gt_list)


    def load_test_data(self, with_gt=False):
        self.create_images_list(self.test_dir)
        self.load_images(self.test_dir)
        self.img_numpy = self.get_data_numpy(self.img_sitk, self.img_list, resample=False, crop=False, normalize=False)
        if with_gt:
            self.create_ground_truths_list()
            self.load_ground_truths(self.test_dir)
            self.gt_numpy = self.get_data_numpy(self.gt_sitk, self.gt_list, resample=False, crop=False, normalize=False)
 
    def create_images_list(self, data_dir):
        # Get list of annotated volumes
        self.img_list = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and 'segmentation' not in f
                         and 'raw' not in f and 'itksnap' not in f]
        # Get list of volumes for which annotations are available
        self.img_list = np.sort(self.img_list)
        print('IMG FILE LIST: ' + str(self.img_list) + '\n')

    def create_ground_truths_list(self):
        for f in self.img_list:
            filename, ext = os.path.splitext(f)
            filename, ext = os.path.splitext(filename)
            # self.gt_list.append(join(filename + '_segmentation' + ext))
            self.gt_list.append(filename + '_segmentation.nii.gz')
        print('GT FILE LIST: ' + str(self.gt_list) + '\n')

    def load_images(self, data_dir):
        """
        Loads images in ITK format from disk and performs the following pre-processing steps:
        rescale intensity: rescale image intensities to a 0-1 range.
        :param data_dir: data directory with data volumes in ITK format.
        :return:
        """
        #rescale_filter = sitk.RescaleIntensityImageFilter()
        #rescale_filter.SetOutputMaximum(self.config_data['n_labels'])
        #rescale_filter.SetOutputMinimum(0)
        rescale_filter = sitk.IntensityWindowingImageFilter()
        rescale_filter.SetOutputMaximum(1)
        rescale_filter.SetOutputMinimum(-1)
        rescale_filter.SetWindowMaximum(240)
        rescale_filter.SetWindowMinimum(-160)

        stats = sitk.StatisticsImageFilter()
        m = 0.
        for f in self.img_list:
            # print("Image name in load_images ", f)
            self.img_sitk[f] = rescale_filter.Execute(
                                sitk.Cast(sitk.ReadImage(os.path.join(data_dir, f)), sitk.sitkFloat32))
            stats.Execute(self.img_sitk[f])
            m += stats.GetMean()
        self.mean_intensity_train = m/len(self.img_sitk)

    def load_ground_truths(self, data_dir):
        for f in self.gt_list:
            # print("Ground truth name in load_ground_truths ", f)
            # self.gt_sitk[f] = sitk.Cast(sitk.ReadImage(join(data_dir, f)) > 0.5, sitk.sitkFloat32)
            self.gt_sitk[f] = sitk.Cast(sitk.ReadImage(os.path.join(data_dir, f)), sitk.sitkFloat32)

    def get_data_numpy(self, data_sitk, data_list, resample=False, crop=False, normalize=False):
        """
        Converts sitk data into numpy arrays, and performs the following pre-processing steps:
        smooth: STILL TO BE IMPLEMENTED smoothing of data before resampling to avoid resampling artifacts
        :param data_sitk: list of data volumes in ITK format
        :param resample: resample images and ground truths to resolution config_data['grid_resolution']
        :param crop: crop images and ground truths to dimensions config_data['grid_size'], around the center of the originals
        :param normalize: normalize image numpy arrays to mean = 0 and a standard deviation = 1
        :return: list of data volumes as numpy arrays
        """
        data_numpy = dict()
        for f in data_list:
            # print("f in get_data_numpy ", f)
            data_numpy[f] = np.zeros(self.grid_size, dtype=np.float32)
            data = data_sitk[f]

            if(resample):
                data = self.resample(data)
            if(crop):
                data = self.crop(data)
            self.img_crop[f] = data

            data_numpy[f] = np.transpose(sitk.GetArrayFromImage(data).astype(dtype=float),
                                           [2, 1, 0])[:, :, :, np.newaxis]
            if(normalize):
                data_numpy[f] = self.normalize(data_numpy[f])

        return data_numpy

    def discrete_gt_label(self, data_numpy, data_list):

        for f in data_list:
            # print("f in bin_gt_label ", f)
            data = data_numpy[f]
            data_numpy[f] = np.round(data)

        return data_numpy

    def resample(self, data, method=sitk.sitkLinear):
        # we rotate the image according to its transformation using the direction and
        # according to the final spacing we want
        # factor = np.asarray(data.GetSpacing()) / self.config_data['grid_resolution']
        spacing = list(data.GetSpacing())
        size = list(data.GetSize())
        factor = [None] * 3
        self.new_size = [None] * 3
        for d in range(3):
            factor[d] = spacing[d] / self.grid_resolution[d]
            self.new_size[d] = max(int(size[d] * factor[d]), self.grid_size[d])

        T = sitk.AffineTransform(3)
        T.SetMatrix(data.GetDirection())
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(data)
        resampler.SetOutputSpacing(self.grid_resolution)
        resampler.SetSize(self.new_size)
        resampler.SetInterpolator(method)
        if False: # self.config['normDir :
            resampler.SetTransform(T.GetInverse())

        data_resampled = resampler.Execute(data)

        return data_resampled

    def crop(self, data):
        data_centroid = [None] * 3
        data_start_pxl = [None] * 3
        for d in range(3):
            data_centroid[d] = int(self.new_size[d] / 2)
            data_start_pxl[d] = int(data_centroid[d] - self.grid_size[d] / 2)
        region_extractor = sitk.RegionOfInterestImageFilter()
        region_extractor.SetSize(self.grid_size)
        region_extractor.SetIndex(data_start_pxl)

        data_cropped = region_extractor.Execute(data)

        return data_cropped

    def normalize(self, data):
        mean = np.mean(data, axis=(0, 1, 2))
        std = np.std(data, axis=(0, 1, 2))
        data_norm = (data - mean[:, np.newaxis, np.newaxis, np.newaxis])/std[:, np.newaxis, np.newaxis, np.newaxis]

        return data_norm
