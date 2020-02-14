#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:13:11 2019

@author: avelinojaver
"""
import numpy as np
import cv2
import random
import math
import torch

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class TransformBase(object):
    def _apply_transform(self, image, target, *args, **argkws):
        image = self._from_image(image, *args, **argkws)
        
        if 'skels' in target:
            skels = target['skels']
            N, S, C = skels.shape
            skels = self._from_coordinates(skels.reshape(-1, 2), *args, **argkws)
            target['skels'] = skels.reshape(N, S, C)
        
        if 'widths' in target:
            target['widths'] = self._from_scalar(target['widths'], *args, **argkws)
        
        return image, target
    
    def _from_coordinates(self, coordinates, *args, **argkws):
        return coordinates
    
    def _from_scalar(self, scalars, *args, **argkws):
        return scalars
        
    def _from_image(self, image, *args, **argkws):
        return image

class AffineTransformBounded(TransformBase):
    def __init__(self, 
                 zoom_range = (0.9, 1.1), 
                 rotation_range = (-90, 90), 
                 crop_size_lims = None,
                 border_mode = cv2.BORDER_CONSTANT,
                 interpolation = cv2.INTER_LINEAR
                 ):
        
        assert not ((zoom_range is None) and (crop_size_lims is None)), "`zoom_range` and `crop_size_lims` cannot be None simultanously"
        
        self.zoom_range = zoom_range
        self.crop_size_lims = crop_size_lims
        
        self.rotation_range = rotation_range
        
        self.w_affine_args = dict(borderMode = border_mode, flags=interpolation)

    
    def __call__(self, image, target):
        theta = np.random.uniform(*self.rotation_range)
        
        if self.crop_size_lims is not None:
            _min_zoom = self.crop_size_lims[0]/min(image.shape)
            _max_zoom = self.crop_size_lims[1]/max(image.shape)
        else:
            _min_zoom, _max_zoom = self.zoom_range
        _zoom_range = (_min_zoom, _max_zoom) if self.zoom_range is None else self.zoom_range
        
            
        _zoom = random.uniform(*_zoom_range)
        _zoom = max(_min_zoom, _zoom)
        _zoom = min(_max_zoom, _zoom)
        
        #https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
     
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), theta, _zoom)
        cos_val = np.abs(M[0, 0])
        sin_val = np.abs(M[0, 1]) #this value is already scaled in the rotation matrix
     
        # compute the new bounding dimensions of the image
        nW = int(((h * sin_val) + (w * cos_val)))
        nH = int(((h * cos_val) + (w * sin_val)))
        
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        return self._apply_transform(image, target, M, img_shape = (nW, nH), zoom = _zoom)
    
    def _from_coordinates(self, coords, M, img_shape = None, zoom = None):
        coords = np.dot(M[:2, :2], coords.T)  +  M[:, -1][:, None]
        coords = coords.T
        return coords
    
    def _from_scalar(self, scalars, *args, zoom = None, **argkws):
        return zoom*scalars
        
    def _from_image(self, image, M, img_shape = None, zoom = None):
        img_shape = image.shape[:2] if img_shape is None else img_shape
        if image.ndim == 2:
            image_rot = cv2.warpAffine(image, M, img_shape, **self.w_affine_args)
        else:
            image_rot = [cv2.warpAffine(image[..., n], M, img_shape, **self.w_affine_args) for n in range(image.shape[-1])] 
            image_rot = np.stack(image_rot, axis=2)
        return image_rot

class PadToSize(TransformBase):
    def __init__(self, roi_size):
        self.roi_size = roi_size
    
    
    
    def __call__(self, image, target):
        l_max = max(image.shape)
        if l_max > self.roi_size:
            scale_factor = self.roi_size/l_max
            target_shape = [round(x*scale_factor) for x in image.shape]
        else:
            scale_factor = None
            target_shape = image.shape
        
        dd = [self.roi_size - x for x in target_shape]
        pad_size = [(math.floor(x/2), math.ceil(x/2)) for x in dd]
        
        image, target = self._apply_transform(image, target, scale_factor, pad_size)
        if not (max(image.shape) <= self.roi_size):
            import pdb
            pdb.set_trace()
            
        assert image.shape == (self.roi_size, self.roi_size)
        return image, target
    
    @staticmethod
    def _from_image(image, scale_factor, pad_size):
        if scale_factor is not None:
            image = cv2.resize(image, dsize = (0,0), fx = scale_factor, fy = scale_factor)
        image = np.pad(image, pad_size, 'constant', constant_values = 0)
        return image
    
    @staticmethod
    def _from_coordinates(coords, scale_factor, pad_size):
        
        corner = np.array([x[0] for x in pad_size[::-1]])[None, None] #the coordinates and padding order are oposite (x,y) corresponds to (dim2,dim1)
        if scale_factor is not None:
            coords = coords*scale_factor
        
        coords = coords + corner
        return coords
    
    @staticmethod
    def _from_scalar(scalars, scale_factor, pad_size):
        if scale_factor is None:
            return scalars
        else:
            return scale_factor*scalars

class RandomVerticalFlip(TransformBase):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            return self._apply_transform(image, target, image.shape)
            
        return image, target
    
    @staticmethod
    def _from_coordinates(coords, img_shape):
        coords_out = coords.copy()
        coords_out[:, 1] = (img_shape[0] - 1 - coords_out[:, 1])
        return coords_out
        
    
    @staticmethod
    def _from_image(image, *args, **argkws):
        return image[::-1]
    
    
class RandomHorizontalFlip(TransformBase):
    def __init__(self, prob = 0.5):
        self.prob = prob
    
    def __call__(self, image, target):
       
        if random.random() < self.prob:
            return self._apply_transform(image, target, image.shape)
        
        return image, target
    
    
    @staticmethod
    def _from_coordinates(coords, img_shape):
        coords_out = coords.copy()
        coords_out[:, 0] = (img_shape[1] - 1 - coords_out[:, 0])
        return coords_out
        
    
    @staticmethod
    def _from_image(image, *args, **argkws):
        return image[:, ::-1]


class NormalizeIntensity(object):
    def __init__(self, scale = (0, 255), norm_mean = 0., norm_sd = 1.):
        self.scale = scale
        self.norm_mean = norm_mean
        self.norm_sd = norm_sd
        
    def __call__(self, image, target):
        
        image = (image.astype(np.float32) - self.scale[0])/(self.scale[1] - self.scale[0])
        
        return image, target


class RandomIntensityOffset(object):
    def __init__(self, offset_range):
        self.offset_range = offset_range
    
    def __call__(self, image, target):
        if random.random() > 0.5:
            if self.offset_range is None:
                vv = image[image!=0]
                rr = np.min(vv), np.max(vv)
                offset = random.uniform(-rr[0], 1 - rr[1])
            else:
                offset = random.uniform(*self.offset_range)
                
            image = image + offset
                
        return image, target


class RandomIntensityExpansion(object):
    def __init__(self, expansion_range = (0.7, 1.3)):
        self.expansion_range = expansion_range
    
    def __call__(self, image, target):
        if self.expansion_range is not None and random.random() > 0.5:
            if image.ndim == 2:
                factor = random.uniform(*self.expansion_range)
            else:
                factor = np.random.uniform(*self.expansion_range, 3)[None, None, :]
                
            image = image*factor
            
        return image, target    

class AddBlankPatch():
    def __init__(self, frac_range = (0.125, 0.25), prob = 0.5):
        self.prob = prob
        self.frac_range = frac_range
    
    @staticmethod
    def _gauss_random_loc(_size, patch_size):
            b = random.gauss(0, 0.4)/2 + 0.5
            #clip
            b = max(0, min(1, b))
            #scale
            b *= _size - patch_size
            return int(round(b))
    
    def __call__(self, image, target):
        if random.random() < self.prob:
        
            patch_dat = []
            for d in image.shape:
                frac = random.uniform(*self.frac_range)
                patch_size = max(1, int(d*frac))
                
                patch_loc = self._gauss_random_loc(d, patch_size)
                patch_dat.append((patch_loc, patch_size))
            
            (x,w), (y,h) = patch_dat
            
            image[x:x+w, y:y+h] = random.random()
        
        return image, target

class AddRandomLine():
    def __init__(self, prob = 0.5):
        self.prob = prob
    
    
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            pt1, pt2 = [tuple([random.randint(0, r-1) for r in image.shape]) for _ in range(2)]
            
            thinkness = random.randint(1, 5)
            
            cv2.line(image, pt1, pt2, random.random(), thinkness)
            
        
        return image, target

class JpgCompression():
    def __init__(self, quality_range = (40, 90), prob = 0.5):
        self.prob = prob
        self.quality_range = quality_range
    
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            quality = random.uniform(*self.quality_range)
            
            img_uint8 = (np.clip(image, 0, 1)*255).astype(np.uint8)
            
            im_encoded = cv2.imencode('.jpg', img_uint8, (cv2.IMWRITE_JPEG_QUALITY, quality))[-1]
            im_decoded = cv2.imdecode(im_encoded, cv2.IMREAD_UNCHANGED)
            
            image = im_decoded.astype(np.float32)/255
        
        return image, target

class RandomFlatField(object):
    def __init__(self, roi_size, prob = 0.5):
        self.prob = prob
        self.roi_size = roi_size
        self.base_factor_range = (0.2, 0.5)
        self.sigma_range = roi_size//4, roi_size
        self.xx, self.yy = np.meshgrid(np.arange(roi_size), np.arange(roi_size))
        
    def __call__(self, image, target):
        
        if  random.random() < self.prob:
            base_factor = random.uniform(*self.base_factor_range)
            
            sigma = random.uniform(*self.sigma_range )
            
            mu_x = random.uniform(0, self.roi_size)
            mu_y = random.uniform(0, self.roi_size)
            dx = self.xx-mu_x
            dy = self.yy-mu_y
            
            flat_field = np.exp(-(dx*dx + dy*dy)/sigma**2)
            bot = flat_field.min()
            
            flat_field -= bot
            flat_field /= (1-bot)
            
            
            flat_field = (1 - base_factor) * flat_field + base_factor
            image = image*flat_field
            
        return image, target
 
class ToTensor(object):
    def __call__(self, image, target):
        image = torch.from_numpy(image).float()
        
        if 'skels' in target:
            target['skels'] = torch.from_numpy(target['skels']).long()
        
        if 'heads' in target:
            target['heads'] = torch.from_numpy(target['heads']).long()
        
        if 'PAF' in target:
            target['PAF'] = torch.from_numpy(target['PAF']).float()
        
        if 'keypoints' in target:
            target['keypoints'] = tuple([torch.from_numpy(x).float() for x in target['keypoints']])
        
        if 'boxes' in target:
            target['boxes'] = torch.from_numpy(target['boxes']).float()
        
        if 'labels' in target:
            target['labels'] = torch.from_numpy(target['labels']).float()
            
        if 'is_valid_ht' in target:
            target['is_valid_ht'] = torch.from_numpy(target['is_valid_ht']).bool()
            
        return image, target