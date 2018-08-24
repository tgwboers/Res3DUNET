import numpy as np
import keras


class TestManager(object):
    """
    Module that returns the predicted labeled images for test images in the class data_loader.
    If 'with_gt' is true it also returns the dice score.
    """
    def __init__(self):
        pass

    def test_model(self, data_loader, model, dice_score, n_labels, with_gt):
        """
        :param model: model that generates the predictions y_pred
        :param dice_score: function that calculates dice score for input y_true, y_pred)
        :param n_labels: number of labels
        """
        self.model = model
        self.dice_score = dice_score
        self.n_labels = n_labels
        test_pred = dict()
        test_gt = dict()
        test_dice = dict()
        for test_name, gt_name in zip(data_loader.img_list, data_loader.gt_list):
            print("Testing image ", test_name)
            test_img = data_loader.img_numpy[test_name]
            test_pred[test_name] = self.model.predict(np.expand_dims(test_img, axis=0))
            if with_gt:
                test_gt[test_name] = np.expand_dims(data_loader.gt_numpy[gt_name], axis=0)
                test_shape = test_gt[test_name].shape[:-1]
                test_gt_categorical = keras.utils.to_categorical(test_gt[test_name], num_classes=self.n_labels+1)\
                    .reshape(test_shape + (self.n_labels + 1,))
                test_dice[test_name] = self.dice_score(test_gt_categorical, test_pred[test_name])

        return test_pred, test_gt, test_dice
        
    




