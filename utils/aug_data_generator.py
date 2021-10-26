import random

import cv2
import numpy as np

from keras_preprocessing.image import ImageDataGenerator


class AugmentedDataGenerator(ImageDataGenerator):
    def __init__(self, per_channel_shift_range=None, swap_channels=False, grayscale=False, **kwargs):
        '''
        Image data generator with custom augmentations
        swap_channels - randomly swap image channels
        grayscale - apply grayscale
        per_channel_shift_range - randomly scale channels independently to imitate white balance shifts, recommended
        value is 50 or below
        '''
        self.swap_channels = swap_channels
        self.grayscale = grayscale
        self.per_channel_shift_range = per_channel_shift_range
        super().__init__(**kwargs)

    def get_random_transform(self, img_shape, seed=None):
        transform_parameters = super().get_random_transform(img_shape, seed)
        if self.swap_channels:
            transform_parameters['channel_swap_idx'] = np.random.choice([0,1,2], 3, replace=False)
        if self.per_channel_shift_range is not None:
            transform_parameters['channel_shift_intensities'] = np.random.uniform(-self.per_channel_shift_range, self.per_channel_shift_range,
                                                                                  3)
        return transform_parameters

    def apply_transform(self, x, transform_parameters):
        x = super().apply_transform(x, transform_parameters)
        if transform_parameters.get('channel_swap_idx') is not None:
            x = self.channel_swap(x, transform_parameters['channel_swap_idx'])
        if transform_parameters.get('channel_shift_intensities') is not None:
            x = self.apply_per_channel_shift(x, transform_parameters['channel_shift_intensities'])
        if self.grayscale:
            x = self.grayscale_transform(x)
        return x

    def grayscale_transform(self, x):
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        return x


    def apply_per_channel_shift(self, x, intensity):
        """ Performs a channel shift independently for each channel to imitate white balance shifts
        """
        channel_axis = self.channel_axis - 1
        x = np.rollaxis(x, channel_axis, 0)
        min_x, max_x = np.min(x), np.max(x)
        channel_images = [
            np.clip(x_channel + intensity[i],
                    min_x,
                    max_x)
            for i, x_channel in enumerate(x)]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x

    def channel_swap(self, x, new_idx):
        # no batch dim here
        channel_axis = self.channel_axis - 1
        # make channel axis first
        x = np.rollaxis(x, channel_axis , 0)
        # randomly reorder channels
        x = x[new_idx, ...]
        # move channel axis to original index
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x
