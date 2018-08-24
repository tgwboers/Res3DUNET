import sys
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import BatchNormalization, ZeroPadding3D, Conv3D, MaxPooling3D, UpSampling3D, Cropping3D, Activation, Add
from keras.optimizers import Adam
from keras import regularizers
import tensorflow as tf
try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate
    
    
class UnetBuilder(object):
    """
    Builds a UNet Keras model with build_model method.
    """

    def __init__(self,config):


        self.input_shape=eval(config.get('DATA', 'grid_size'))
        self.pool_size=eval(config.get('MODEL', 'pool_size'))
        self.initial_learning_rate=config.getfloat('TRAINING', 'initial_learning_rate')

        self.n_labels = config.getint('DATA', 'n_labels')
        self.wt_fac = eval(config.get('TRAINING', 'loss_weight_factors'))
        self.wt_fac = self.wt_fac[:self.n_labels+1]

    def build_model(self):

        
        """
        #:param input_shape: Shape of the input data (x_size, y_size, z_size, n_channels).
        #:param padding: padding applied to images
        #:param downsize_filters_factor: Factor to which to reduce the number of filters.
        #Making this value larger will reduce the amount of memory the model will need during training.
        #:param pool_size: Pool size for the max pooling operations.
        #:param n_labels: Number of binary labels that the model is learning.
        #:param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
        #:param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsampling.
        #This increases the amount memory required during training.
        #:return: Untrained 3D UNet Model
        """

        def Conv3D_BN(filters, ks, activation, padding, kernel_initializer, input):

            conv3d = Conv3D(filters, ks, activation=None, padding=padding, use_bias=False,
                            kernel_initializer=kernel_initializer)(input)
            conv3d_bn = BatchNormalization(axis=4, beta_initializer='zeros', gamma_initializer='ones')(conv3d)
            conv3d_out = Activation(activation)(conv3d_bn)
            return conv3d_out

        def bridge(layer1, layer2):
            # Defines bridge between layers of the same resolution from the downsampling and form the upsampling paths.
            crop = np.ceil(np.subtract(layer1.shape.as_list()[1:4], layer2.shape.as_list()[1:4]) / 2).astype(int)
            if all(layer2.shape.as_list()[1:4] + 2 * crop == layer1.shape.as_list()[1:4]):
                cropping = tuple(map(tuple, np.transpose([crop, crop])))
            else:
                sys.exit("Cropping at " + layer1.name + "-" + layer2.name + "bridge is not symmetric, "
                                                                            "fix image and input dimensions to fit Unet structure.")
            layer2 = concatenate([layer2, Cropping3D(cropping=cropping)(layer1)], axis=4)

            return layer2

        input_shape_img = self.input_shape + [1]
        inputs_img = Input(input_shape_img, dtype='float32') # 60 60 18


        conv1p = ZeroPadding3D(padding=(0,0,0))(inputs_img) # 64 64 24
        conv1a = Conv3D(64, (3, 3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(0), kernel_initializer='glorot_uniform')(conv1p)
        conv1b = Conv3D_BN(64, (3, 3, 1), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', input=conv1a)
        conv1c = Add()([conv1a, conv1b])
        pool1 = MaxPooling3D(pool_size=self.pool_size)(conv1c) # 32 32 12
        conv2a = Conv3D_BN(128, (3, 3, 3), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', input=pool1)
        conv2b = Conv3D_BN(128, (3, 3, 1), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', input=conv2a)
        conv2c = Add()([conv2a, conv2b])
        pool2 = MaxPooling3D(pool_size=self.pool_size)(conv2c) # 16 16 6
        conv3a = Conv3D_BN(256, (3, 3, 3), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', input=pool2)
        conv3b = Conv3D_BN(256, (3, 3, 1), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', input=conv3a)
        conv3c = Add()([conv3a, conv3b])
        pool3 = MaxPooling3D(pool_size=self.pool_size)(conv3c) # 8 8 3
        conv4a = Conv3D_BN(512, (3, 3, 3), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform',  input=pool3)
        conv4b = Conv3D_BN(512, (3, 3, 1), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', input=conv4a)
        conv4c = Add()([conv4a, conv4b])

        # upsampling
        up5a = UpSampling3D(size=self.pool_size)(conv4c)
        up5b = bridge(conv3c, up5a)
        conv5a = Conv3D_BN(256, (3, 3, 3), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', input=up5b)
        conv5b = Conv3D_BN(256, (3, 3, 1), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', input=conv5a)
        conv5c = Add()([conv5a, conv5b])
        up6a = UpSampling3D(size=self.pool_size)(conv5c)
        up6b = bridge(conv2c, up6a)
        conv6a = Conv3D_BN(128, (3, 3, 3), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', input=up6b)
        conv6b = Conv3D_BN(128, (3, 3, 1), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', input=conv6a)
        conv6c = Add()([conv6a, conv6b])
        up7a = UpSampling3D(size=self.pool_size)(conv6c)
        up7b = bridge(conv1c, up7a)
        conv7a = Conv3D_BN(64, (3, 3, 3), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', input=up7b)
        conv7b = Conv3D_BN(64, (3, 3, 1), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', input=conv7a)
        conv7c = Add()([conv7a, conv7b])

        # Final softmax layer for class probabilities for each input voxel
        conv8 = Conv3D(self.n_labels+1, (1, 1, 1))(conv7c)  # include background label

        act = Activation('softmax', name='seg')(conv8)

        model = Model(inputs=inputs_img, outputs=act)


        # Metrics
        metrics = []
        for label in range(self.n_labels):
            dice_coef_c = self.create_dice_coef(label)
            dice_coef_c.__name__ = 'dice_coef_lbl' + str(label+1)
            metrics.append(dice_coef_c)


            # Add optimizer and loss function
            model.compile(optimizer=Adam(lr=self.initial_learning_rate),
                                       loss={'seg': self.weighted_categorical_crossentropy},
                                       loss_weights={'seg': 1.},
                                       metrics={'seg':metrics})

        return model

    def create_dice_coef(self,label):
        def dice_coef_c(y_true, y_pred, smooth=1.):
            y_true_f=K.flatten(y_true[:,:,:,:,label+1])
            y_pred_f=K.flatten(y_pred[:,:,:,:,label+1])
            intersection= K.sum(y_true_f * y_pred_f)
            dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

            return dice
        return dice_coef_c

    def weighted_categorical_crossentropy(self, y_true, y_pred):
        # Weighted categorical cross entropy is used as loss function, because the classes are imbalanced.
        # This function is not available for 3D data, in Keras, and is defined here based on the code of the
        # Keras function 'categorical_crossentropy'.
        _EPSILON = 10e-8
        # Scale predictions so that the class probabilities of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred,
                                axis=len(y_pred.get_shape()) - 1,
                                keep_dims=True)
        # Ensuring probabilities <0, 1>
        epsilon = tf.cast(_EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        # Weights used in loss function
        wt_fac = self.wt_fac
        wt_map = y_true * wt_fac
        loss_map = wt_map * y_true * tf.log(y_pred)
        # Manual computation of crossentropy
        loss = - tf.reduce_sum(loss_map, axis=len(y_pred.get_shape()) - 1)
        return loss



