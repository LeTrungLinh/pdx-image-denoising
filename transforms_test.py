import random
import torch
import numpy as np
import cv2
import albumentations as A
import random
from PIL import Image
from PIL import ImageEnhance
import imgaug.augmenters as iaa
class AdditiveWhiteGaussianNoise(object):
    """Additive white gaussian noise generator."""
    def __init__(self, noise_level, fix_sigma=False, clip=False):
        self.noise_level = noise_level
        self.fix_sigma = fix_sigma
        self.rand = np.random.RandomState(1)
        self.clip = clip
        if not fix_sigma:
            self.predefined_noise = [i for i in range(5, noise_level + 1, 5)]

    def __call__(self, sample):
        """
        Generates additive white gaussian noise, and it is applied to the clean image.
        :param sample:
        :return:
        """
        image = sample.get('image')

        # rand_list = random.sample(range(1, 4), 1) 
        rand_list = [2]
        if rand_list[0] == 1:
           seq = iaa.Sequential([
                iaa.pillike.EnhanceColor((0.5,0.8)),
                iaa.Multiply((0.5, 1.5))
            ])
           
        if rand_list[0] == 2:
            seq = iaa.Sequential([
                iaa.JpegCompression(compression=(70, 90)),
                iaa.LinearContrast([0.5,1.5])  
            ])
            
        if rand_list[0] == 3:
           seq = iaa.Sequential([
                iaa.MotionBlur(k=5, angle=[-15, 15])
            ]) 
           
        if rand_list[0] == 4:
           seq = iaa.Sequential([
                iaa.AdditivePoissonNoise(50, per_channel=True)
            ])

        # Apply augmentations
        noisy = seq.augment_images(image)

        if self.clip:
            noisy = np.clip(noisy, 0., 255.)

        
        return {'image': image, 'noisy': noisy.astype('float32')}



class ToTensor(object):
    """Convert data sample to pytorch tensor"""
    def __call__(self, sample):
        image, noisy = sample.get('image'), sample.get('noisy')
        image = torch.from_numpy(image.transpose((2, 0, 1)).astype('float32') / 255.)

        if noisy is not None:
            noisy = torch.from_numpy(noisy.transpose((2, 0, 1)).astype('float32') / 255.)

        return {'image': image, 'noisy': noisy}


class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image, noisy = sample.get('image'), sample.get('noisy')
            image = np.flipud(image)

            if noisy is not None:
                noisy = np.flipud(noisy)

            return {'image': image, 'noisy': noisy}

        return sample


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image, noisy = sample.get('image'), sample.get('noisy')
            image = np.fliplr(image)

            if noisy is not None:
                noisy = np.fliplr(noisy)

            return {'image': image, 'noisy': noisy}

        return sample


class RandomRot90(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image, noisy = sample.get('image'), sample.get('noisy')
            image = np.rot90(image)

            if noisy is not None:
                noisy = np.rot90(noisy)

            return {'image': image, 'noisy': noisy}

        return sample







# class AdditiveWhiteGaussianNoise(object):
#     """Additive white gaussian noise generator."""
#     def __init__(self, noise_level, fix_sigma=False, clip=False):
#         self.noise_level = noise_level
#         self.fix_sigma = fix_sigma
#         self.rand = np.random.RandomState(1)
#         self.clip = clip
#         if not fix_sigma:
#             self.predefined_noise = [i for i in range(5, noise_level + 1, 5)]

#     def __call__(self, sample):
#         """
#         Generates additive white gaussian noise, and it is applied to the clean image.
#         :param sample:
#         :return:
#         """
#         image = sample.get('image')

#         # rand_list = random.sample(range(1, 4), 1) 
#         rand_list = [2]
#         if rand_list[0] == 1:
#             kernel_size = (9, 9) #you can change kernel size according to your needs
#             sigma = 3 #you can change sigma value according to your needs 
#             noisy = np.zeros((image.shape[0], image.shape[1], image.shape[2], image.shape[3]))
#             for i in range(image.shape[0]):
#                 img = image[i,:,:,:]
#                 blurred_img = cv2.GaussianBlur(img, kernel_size, sigma)
#                 noisy[i,:,:,:] = blurred_img
#         if rand_list[0] == 2:
#             # noisy = image*0.5
#             seq = iaa.Sequential([
#                 iaa.pillike.EnhanceColor((0.5,0.8))
#             ])
#             # Apply augmentations
#             noisy = seq.augment_images(image)
#         if rand_list[0] == 3:
#             if image.ndim == 4:                 # if 'image' is a batch of images, we set a different noise level per image
#                 samples = image.shape[0]        # (Samples, Height, Width, Channels) or (Samples, Channels, Height, Width)
#                 if self.fix_sigma:
#                     sigma = self.noise_level * np.ones((samples, 1, 1, 1))
#                 else:
#                     sigma = np.random.choice(self.predefined_noise, size=(samples, 1, 1, 1))
#                 noise = self.rand.normal(0., 1., size=image.shape)
#                 noise = noise * sigma
#             else:                               # else, 'image' is a simple image
#                 if self.fix_sigma:              # (Height, Width, Channels) or (Channels , Height, Width)
#                     sigma = self.noise_level
#                 else:
#                     sigma = self.rand.randint(5, self.noise_level)
#                 noise = self.rand.normal(0., sigma, size=image.shape)
#             noisy = image + noise
#         if rand_list[0] == 4:
#             # Define the size of the kernel
#             kernel_size = 9

#             # Generate the motion kernel
#             kernel_motion_blur = np.zeros((kernel_size, kernel_size))
#             kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
#             kernel_motion_blur = kernel_motion_blur / kernel_size
#             noisy = np.zeros((image.shape[0], image.shape[1], image.shape[2], image.shape[3]))
#             for i in range(image.shape[0]):
#                 img = image[i,:,:,:]
#                 motionblur_image = cv2.filter2D(img, -1, kernel_motion_blur)
#                 noisy[i,:,:,:] = motionblur_image

#         if self.clip:
#             noisy = np.clip(noisy, 0., 255.)

        

#         return {'image': image, 'noisy': noisy.astype('float32')}