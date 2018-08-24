import os
import numpy as np
import pickle
import SimpleITK as sitk


def create_output_folders(config):

    print('generating folders')

    if not os.path.exists(os.path.abspath(config.get('PATH', 'figures_dir'))):
        os.makedirs(os.path.abspath(config.get('PATH', 'figures_dir')))
    if not os.path.exists(os.path.abspath(config.get('PATH',  'hdf5data_dir'))):
        os.makedirs(os.path.abspath(config.get('PATH', 'hdf5data_dir')))
    if not os.path.exists(os.path.abspath(config.get('PATH', 'newmodel_dir'))):
        os.makedirs(os.path.abspath(config.get('PATH', 'newmodel_dir')))
    if not os.path.exists(os.path.abspath(config.get('PATH', 'logging_dir'))):
        os.makedirs(os.path.abspath(config.get('PATH', 'logging_dir')))
    if not os.path.exists(os.path.abspath(config.get('PATH', 'pickle_dir'))):
        os.makedirs(os.path.abspath(config.get('PATH', 'pickle_dir')))
    if not os.path.exists(os.path.abspath(config.get('PATH', 'mhd_dir'))):
        os.makedirs(os.path.abspath(config.get('PATH', 'mhd_dir')))



def write_to_sitk(test_data, config_data, test_prefix, shape, origin, direction, spacing,
                  orig_size, orig_origin, orig_direction, orig_spacing,
                  mhd_format=sitk.sitkFloat32, resample_to_original=False):
    # Write test data and predictions to .mhd/.raw files.
    test_sitk = sitk.Image((shape[1], shape[2], shape[3]), mhd_format, shape[0])
    test_sitk.SetOrigin(origin)
    test_sitk.SetDirection(direction)
    test_sitk.SetSpacing(spacing)
    sitk._SimpleITK._SetImageFromArray(test_data.tostring(), test_sitk)
    if resample_to_original:
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(orig_spacing)
        resampler.SetOutputOrigin(orig_origin)
        resampler.SetOutputDirection(orig_direction)
        resampler.SetSize(orig_size)
        resampler.SetInterpolator(sitk.sitkLabelGaussian)  # sitk.sitkNearestNeighbor)
        test_sitk = resampler.Execute(test_sitk)
    sitk.WriteImage(test_sitk, config_data['mhd_dir'] + test_prefix)


def write_slices_to_sitk(array, filename, datadir):
    shape = array.shape
    dataDir = datadir
    if len(shape) > 3:
        for i in range(shape[3]):
            slice_array = array[:, :, :, i].transpose()
            image = sitk.GetImageFromArray(slice_array)
            fName = dataDir + "/" + filename + "_channel" + str(i) + ".nii"
            sitk.WriteImage(image, fName)
    else:
        slice_array = array.transpose()
        slice_array = np.ascontiguousarray(slice_array)
        image = sitk.GetImageFromArray(slice_array)
        fName = dataDir + "/" + filename + ".nii"
        sitk.WriteImage(image, fName)


def write_predictions_to_sitk(config_data, data_loader, test_pred):
    """
    Writes .mhd volumes of the test image (e.g. "Test_Case00.mhd"), predicted segmentation (e.g. "Pred_Case00.mhd"),
    and difference between predicted segmentation adnd ground truth (e.g. "Diff_Case00.mhd").
    :param config_data:
    :param data_loader:
    :param test_pred: network output prediction
    :param threshold: threshold for segmentation output
    :return:
    """

    for test_name, gt_name in zip(data_loader.img_list, data_loader.gt_list):
        test_pred_label = np.expand_dims(np.argmax(test_pred[test_name], axis=4), axis=4)
        # Get position and orientation from original image
        mhd_shape = test_pred_label.shape
        mhd_origin = data_loader.img_crop[test_name].GetOrigin()
        mhd_direction = data_loader.img_crop[test_name].GetDirection()
        mhd_spacing = data_loader.img_crop[test_name].GetSpacing()
        orig_size = data_loader.img_sitk[test_name].GetSize()
        orig_origin = data_loader.img_sitk[test_name].GetOrigin()
        orig_direction = data_loader.img_sitk[test_name].GetDirection()
        orig_spacing = data_loader.img_sitk[test_name].GetSpacing()
        # Write test input image in Unet resolution
        image = np.float32(
            np.expand_dims(np.transpose(data_loader.img_numpy[test_name][:, :, :, 0], [2, 1, 0]), axis=0))
        write_to_sitk(image, config_data, "/test_" + test_name, mhd_shape, mhd_origin, mhd_direction, mhd_spacing,
                      orig_size, orig_origin, orig_direction, orig_spacing,
                      mhd_format=sitk.sitkFloat32, resample_to_original=False)
        # Write ground truth image in Unet resolution
        ground_truth = np.int32(np.expand_dims(np.transpose(data_loader.gt_numpy[gt_name][:, :, :, 0],
                                                            [2, 1, 0]), axis=0))
        write_to_sitk(ground_truth, config_data, "/gt_" + test_name, mhd_shape, mhd_origin, mhd_direction, mhd_spacing,
                      orig_size, orig_origin, orig_direction, orig_spacing,
                      mhd_format=sitk.sitkUInt32, resample_to_original=False)
        # Write test prediction in Unet and in original resolution
        prediction = np.int32(np.transpose(test_pred_label[:, :, :, :, 0], [0, 3, 2, 1]))
        write_to_sitk(prediction, config_data, "/pred_" + test_name, mhd_shape, mhd_origin, mhd_direction, mhd_spacing,
                      orig_size, orig_origin, orig_direction, orig_spacing,
                      mhd_format=sitk.sitkUInt32, resample_to_original=False)
        write_to_sitk(prediction, config_data, "/pred_orig_" + test_name, mhd_shape, mhd_origin, mhd_direction, mhd_spacing,
                      orig_size, orig_origin, orig_direction, orig_spacing,
                      mhd_format=sitk.sitkUInt32, resample_to_original=True)
        # Write difference between prediction and ground truth in Unet resolution
        difference = np.zeros(prediction.shape, dtype=np.int32)
        difference[np.where((prediction == 2) & (ground_truth != 2))] = 4
        difference[np.where((prediction != 2) & (ground_truth == 2))] = 3
        difference[np.where((prediction == 1) & (ground_truth != 1))] = 6
        difference[np.where((prediction != 1) & (ground_truth == 1))] = 5
        write_to_sitk(difference, config_data, "/diff_" + test_name, mhd_shape, mhd_origin, mhd_direction, mhd_spacing,
                      orig_size, orig_origin, orig_direction, orig_spacing,
                      mhd_format=sitk.sitkUInt32, resample_to_original=False)
        write_to_sitk(difference, config_data, "/diff_orig_" + test_name, mhd_shape, mhd_origin, mhd_direction, mhd_spacing,
                      orig_size, orig_origin, orig_direction, orig_spacing,
                      mhd_format=sitk.sitkUInt32, resample_to_original=True)


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def hist_match(source, template):
    """
    https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    #interp_t_values = np.zeros_like(source,dtype=float)
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)