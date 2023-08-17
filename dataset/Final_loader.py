#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:10:33 2017

@author: yanrpi
"""

# %%
import os
import glob
import numpy as np
import nibabel as nib
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from os import path
# from scipy.misc import imsave
from scipy import ndimage
import skimage
import cv2

from PIL import Image, ImageFilter
import torchvision

class PartialDataset(Dataset):
    def __init__(self, root_dir, transform = None, type = 'nii'):

        self.root_dir = root_dir #根目录
        self.transform = transform #图像转换

        self.filenames = glob.glob(path.join(root_dir, '*.{}'.format(type))) #数据集图像列表
        print(path.join(root_dir, '*.{}').format(type))
        
        self.num_images = len(self.filenames) #数据集大小
        print('num_images = ', self.num_images)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx): #获取数据
        t1 = time.time()
        img_name = self.filenames[idx] #数据名列表
        #print(img_name.split('/')[-1:], flush = True)
        #根据图像数据路径，替换其中部分字符来得到标签路径

        if 'LiTS' in img_name:
            if 'unlabeled' in img_name: #未标注数据，数据自身路径作为标签路径，后续不使用
                seg_name = img_name
            elif 'validation' in img_name: #验证集
                seg_name = img_name.replace('validation', 'label')
                seg_name = seg_name.replace('volume', 'segmentation')
            else: #标注数据
                seg_name = img_name.replace('labeled', 'label')
                seg_name = seg_name.replace('volume', 'segmentation')    
                
        elif 'KiTS' in img_name:
            if 'unlabeled' in img_name: #未标注数据，数据自身路径作为标签路径，后续不使用
                #seg_name = img_name  

                seg_name = img_name.replace('new_unlabeled', 'label_2') #real
                #seg_name = img_name.replace('unlabeled', 'label')
                seg_name = seg_name.replace('scan', 'label')
                
            elif 'validation' in img_name: #验证集
                seg_name = img_name.replace('new_validation', 'label_2') #real
                #seg_name = img_name.replace('validation', 'label')
                seg_name = seg_name.replace('scan', 'label')
            else: #标注数据
                seg_name = img_name.replace('new_labeled', 'label_2') #real
                #seg_name = img_name.replace('labeled', 'label')
                seg_name = seg_name.replace('scan', 'label')    
                
        elif 'Spleen' in img_name:
            if 'unlabeled' in img_name: #未标注数据，数据自身路径作为标签路径，后续不使用
                #seg_name = img_name
                seg_name = img_name.replace('unlabeled', 'label_3')
                
                
            elif 'validation' in img_name: #验证集
                seg_name = img_name.replace('validation', 'label_3')
                #seg_name = img_name.replace('validation', 'label')
            else: #标注数据
                seg_name = img_name.replace('labeled', 'label_3')
                #seg_name = img_name.replace('labeled', 'label')
                
        image = nib.load(img_name).get_data() #读入图像数据
        segmentation = nib.load(seg_name).get_data() #读入标注数据
        
        #print('image.shape = {} seg.shape = {}'.format(image.shape, segmentation.shape))
        #都是 256 * 256 * xxx
        t2 = time.time()
        if 'unlabeled' in img_name: #未标记数据，同时生成强增强数据s1和s2
            '''
            image_s1 = image
            image_s2 = image
            '''
            sample = {'image': image, 'label': segmentation}

            '''
            sample2 = {'image': image_s1, 'label': segmentation}
            sample3 = {'image': image_s2, 'label': segmentation}
            '''
            
            
            if self.transform: #图像转换
                sample = self.transform(sample)
                '''
                sample2 = self.transform(sample2)
                sample3 = self.transform(sample3)
                '''
                #RandomCrop((hw, hw, slices),view) → Clip(-200, 200) → Normalize(-200, 200) → RandomHorizontalFlip() → RandomVerticalFlip() → ToTensor()
                #随机裁剪 → 固定数据上下界 → ？ → 随机水平翻转 → 随机垂直翻转 →
                
            img = sample['image']
            #img_s1 = sample2['image']
            #img_s2 = sample3['image']
            #label = sample1['label']

            #print('img.shape = {} img_s1.shape = {} img_s2.shape = {} label.shape = {}'.format(img.shape, img_s1.shape, img_s2.shape, label.shape))
            #都是 3 * 224 * 224
            '''
            slice = np.array(img[0:1, :, :].squeeze(dim = 0).cpu())
            print('slice.shape = ', slice.shape)
            name = './img/x/{}.png'.format(img_name.split('/')[-1:])
            print('name = ', name)
            slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
            slice.save(name)
            '''
            #弱扰动（随机90°旋转+随机角度旋转）+强扰动（颜色扰动+灰度变化+高斯模糊）
            
            
            img_w = WeakDisturb(img)
            img_s1 = StrongDisturb(img_w)
            img_s2 = StrongDisturb(img_w)
            
            '''
            img_w = img#WeakDisturb(img)
            img_s1 = img_w
            img_s2 = img_w
            '''
            
            '''
            slice = np.array(img[0:1, :, :].squeeze(dim = 0).cpu())
            print('slice.shape = ', slice.shape)
            name = './img/w/{}.png'.format(img_name.split('/')[-1:])
            print('name = ', name)
            slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
            slice.save(name)
            '''
            
            #print('img_w.shape = {} label_w.shape = {}'.format(img.shape, label.shape)) #[3, 224, 224] [1, 224, 224]
            #img_s1 = WeakDisturb(img_s1, label, idx)
            #img_s1 = StrongDisturb(img, label, idx)
            '''
            slice = np.array(img_s1[0:1, :, :].squeeze(dim = 0).cpu())
            print('slice.shape = ', slice.shape)
            name = './img/s/{}.png'.format(img_name.split('/')[-1:])
            print('name = ', name)
            slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
            slice.save(name)
            '''
            
            #print('img_s1.shape = {} label_s1.shape = {}'.format(img_s1.shape, label.shape)) #[3, 224, 224] [1, 224, 224]
            #img_s2, label = WeakDisturb(img_s2, label, idx)
            #img_s2, label = StrongDisturb(img, label, idx)
            
            sample = {'image': img_w, 'image_s1': img_s1, 'image_s2': img_s2} #未标注数据，只返回原始数据+原始强扰动数据*2
            
        else : #标注数据
            sample = {'image': image, 'label': segmentation}
            
            if self.transform: #图像转换
                sample = self.transform(sample)
                
            img = sample['image']
            label = sample['label']

            #弱扰动（随机90°旋转+随机角度旋转）
            #img, label = WeakDisturb(img, label)
            #强扰动（颜色扰动+灰度变化+高斯模糊）
            #img, label = StrongDisturb(img, label)
        t3 = time.time()
        #print('Load time = ', t2 - t1)
        #print('Trans time = ', t3 - t2)
        return sample


def WeakDisturb(img):#, label, idx): #弱扰动（随机90°旋转+随机角度旋转）
    #先转换成numpy.array格式
    img = np.array(img)#.cpu())
    #label = np.array(label.cpu())
    '''
    print('W-start-img.shape = ', img.shape)
    slice = np.array(img[0])
    print('slice.shape = ', slice.shape)
    name = './image_test/W-start/{}.png'.format(idx)
    print('name = ', name)
    slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
    slice.save(name)
    '''
    
    N = img.shape[0] #层数
    k = np.random.randint(0, 4) #旋转次数
    angle = np.random.randint(-20, 20) #旋转角度
    
    for i in range(N):
        img[i] = np.rot90(img[i], k) #图像随机90°旋转
    '''
    slice = np.array(img[0])
    print('slice.shape = ', slice.shape)
    name = './image_test/W-rot90/{}.png'.format(idx)
    print('name = ', name)
    slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
    slice.save(name)
    '''
    #img = ndimage.rotate(img, angle, order = 0, reshape = False) #图像随机角度旋转
    '''
    slice = np.array(img[0])
    print('slice.shape = ', slice.shape)
    name = './image_test/W-rotate/{}.png'.format(idx)
    print('name = ', name)
    slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
    slice.save(name)
    '''
    '''
    label[0] = np.rot90(label[0], k) #标注随机90°旋转


    #再转换回tensor格式
    img = torch.tensor(img).cuda()
    label = torch.tensor(label).cuda()
    '''
    
    '''
    slice = np.array(img[0:1, :, :].squeeze(dim = 0).cpu())
    print('slice.shape = ', slice.shape)
    name = './image_test/W-tensor/{}.png'.format(idx)
    print('name = ', name)
    slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
    slice.save(name)
    '''
    return img#, label

def blur(img, p = 0.5): #强扰动——模糊
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius = sigma))
    return img

def StrongDisturb(img):#, label, idx): #强扰动（颜色扰动+灰度变化+高斯模糊）
    #先转换成numpy.array格式
    img = np.array(img)#.cpu())
    #label = np.array(label.cpu())

    N = img.shape[0] #层数
    cache = [] #暂存经过强扰动处理后的img各层
    '''
    tmp = np.array(img[0])
    print('slice.shape = ', tmp.shape)
    name = './image_test/AA/{}-{}.png'.format(idx, 0)
    print('name = ', name)
    tmp = Image.fromarray((tmp * 255).astype('float32')).convert('L')
    tmp.save(name)
    '''
    
    for i in range(N):
        slice = img[i] #取出当前层，并转换成PIL.Image格式
        slice = Image.fromarray((slice * 255.0).astype('float32')).convert('L')
        '''
        name = './image_test/BB/{}-{}.png'.format(idx, i)
        print('name = ', name)
        slice.save(name)
        '''
        
        #施加强扰动
        if random.random() < 0.8:
            slice = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(slice)
        '''    
        name = './image_test/CC/{}-{}.png'.format(idx, i)
        print('name = ', name)
        slice.save(name)
        '''
        '''
        tmp = np.array(slice)
        print('slice.shape = ', tmp.shape)
        name = './image_test/S-CJ/{}-{}.png'.format(idx, i)
        print('name = ', name)
        tmp = Image.fromarray((tmp * 255).astype('float32')).convert('L')
        tmp.save(name)
        '''
        slice = transforms.RandomGrayscale(p = 0.2)(slice)
        '''
        name = './image_test/DD/{}-{}.png'.format(idx, i)
        print('name = ', name)
        slice.save(name)
        '''
        '''
        tmp = np.array(slice)
        print('slice.shape = ', tmp.shape)
        name = './image_test/S-RG/{}-{}.png'.format(idx, i)
        print('name = ', name)
        tmp = Image.fromarray((tmp * 255).astype('float32')).convert('L')
        tmp.save(name)
        '''
        slice = blur(slice)
        '''
        name = './image_test/EE/{}-{}.png'.format(idx, i)
        print('name = ', name)
        slice.save(name)
        '''
        '''
        tmp = np.array(slice)
        print('slice.shape = ', tmp.shape)
        name = './image_test/S-Blur/{}-{}.png'.format(idx, i)
        print('name = ', name)
        tmp = Image.fromarray((tmp * 255).astype('float32')).convert('L')
        tmp.save(name)
        '''
        #放到cache中准备组合
        cache.append(np.array(slice) / 255.0)
    #将强扰动后的各层图片组合
    cache = np.array(cache)
    img = cache
    '''
    for i in range(N):
        slice = np.array(img[i])
        print('slice.shape = ', slice.shape)
        name = './image_test/EE/{}-{}.png'.format(idx, i)
        print('name = ', name)
        slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
        slice.save(name)
    '''
    
    '''
    #取出标注，并转换成PIL.Image格式
    slice = Image.fromarray(label[0].astype('float32')).convert('L')
    #施加强扰动
    
    if random.random() < 0.8:
        slice = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(slice)
    
    slice = transforms.RandomGrayscale(p = 0.2)(slice)
    slice = blur(slice)
    #将强扰动后的标注导入原标注
    label[0] = np.array(slice)
    '''
    #再转换回tensor格式
    img = torch.tensor(img).cuda()
    #label = torch.tensor(label).cuda()
    return img#, label
    
class RandomCrop(object): #随机裁剪，选出其中标注非零的层数
    """Crop randomly the image in a sample.
    For segmentation training, only crop sections with non-zero label

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, view, dataset):
        assert isinstance(output_size, (int, tuple)) #判断output_size是否为整型/元组
        if isinstance(output_size, int): #为整型时
            self.output_size = (output_size, output_size, output_size)
        else: #为元组时
            assert len(output_size) == 3 
            self.output_size = output_size
        self.view = view
        self.dataset = dataset

    def __call__(self, sample):
        #t1 = time.time()
        
        image, segmentation = sample['image'], sample['label'] #图像和标注
        #print('RandomCrop img.type = {} segmentation.type = {}'.format(type(image), type(segmentation))) #numpy.memmap
        
        if self.dataset != 2 and self.dataset != 2.5:
            h, w, d = image.shape
        else:
            d, h, w = image.shape #数据原尺寸 长*宽*深
            w = 256
        new_h, new_w, new_d = self.output_size #数据新尺寸 长*宽*深
        view = self.view #图像扫描方向
        new_d_half = new_d >> 1 #新尺寸深度的一半

        # Find slices containing segmentation object

        img_data = image #图像
        seg_data = segmentation #标注

        
        '''
        idx = random.randint(0, 100)
        img = img_data.transpose((2, 1, 0))
        C = img.shape[0]
        for i in range(C):
            slice = img[i]
            #print('img[{}].shape = {}'.format(i, img.shape))
            print('slice.shape = ', slice.shape)
            name = './img/1/{}-{}.png'.format(idx, i)
            print('name = ', name)
            slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
            slice.save(name)
        '''
        '''
        print('BEFORE seg.unique = ', np.unique(seg_data))
        print('BEFORE seg.dtype = ', seg_data.dtype)
        '''
        if self.dataset == 2 or self.dataset == 2.5:
            img_data = skimage.transform.resize(img_data, (256, 256, 256))
            seg_data = skimage.transform.resize(seg_data.astype(np.float64), (256, 256, 256)).astype(np.int64)
        '''
        print('AFTER seg.unique = ', np.unique(seg_data))
        print('AFTER seg.dtype = ', seg_data.dtype)
        print('RandomCrop img.shape = ', img_data.shape)
        '''
        #根据图像扫描方向改变数据各维度的顺序        
        if view == 'axial':
            img_data = img_data
            seg_data = seg_data
        elif view == 'coronal':
            img_data = img_data.transpose((2, 0, 1))
            seg_data = seg_data.transpose((2, 0, 1))
        else:
            img_data = img_data.transpose((2, 1, 0))
            seg_data = seg_data.transpose((2, 1, 0))
            
        if self.dataset == 1.5 or self.dataset == 2.5 or self.dataset == 3.5 or self.dataset == 4.5:
            lim = 50
            seg_start = max((d >> 1) - lim, 0) 
            seg_end = min((d >> 1) + lim, d - new_d) 
            #print('U')
        else:
            #print('L')
            if self.dataset == 2 or self.dataset == 2.5:
                summed = np.sum(seg_data.sum(axis = 1), axis = 1) / self.dataset #依次对标注的前两维求和，得出标注的每层元素之和
            else:
                summed = np.sum(seg_data.sum(axis = 0), axis = 0) / self.dataset #依次对标注的前两维求和，得出标注的每层元素之和

            #img_summed = np.sum(img_data.sum(axis = 0), axis = 0)
            
            #print('img.max = {} seg.max = {}'.format(np.argmax(img_summed), np.argmax(summed)))

            
            #print(summed)
            #print('summed_shape = ', summed.shape) #大小与标注的层数一致
            #print('summed.max = {} summed.max->slice = {}'.format(np.max(summed), np.argmax(summed)))
            
            if self.dataset == 1:
                threshold = 3000
            elif self.dataset == 2:
                threshold = 2000
            elif self.dataset == 3:
                threshold = 1000
            
            non0_list = np.asarray([i for i in range(summed.size)])
            non0_list = non0_list[(summed > threshold)] #标注中层内元素之和大于阈值的层的编号，即非空的那些层
            #print('non0_list = [{}, {}] all_slice = {}'.format(np.min(non0_list), np.max(non0_list), d))
            #print('percent = {}'.format(len(non0_list) * 1.0 / (d * 1.0)))
            
            if len(non0_list) == 0:
                lim = 50
                seg_start = max((d >> 1) - lim, 0) 
                seg_end = min((d >> 1) + lim, d - new_d)
                #print('no')
            else:
                #深度一维的下标的合法范围，保证后续作为标注的一层不会超出原深度的范围
                seg_start = max(np.min(non0_list) - new_d_half, 0) 
                seg_end = min(np.max(non0_list) + new_d_half, d - new_d) 
                #print('yes')
            
            '''
            lim = 10
            seg_start = max(np.argmax(summed) - lim, 0) 
            seg_end = min(np.argmax(summed) + lim, d) 
            '''
        if new_h == h: #新尺寸与原尺寸相同，前两维的起始下标从0开始
            top = 0
            left = 0
        else: #新尺寸与原尺寸不同时，前两维的起始下标在合法范围内随机选取
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
        #print('seg_start = {} seg_end = {}'.format(seg_start, seg_end))


        '''
        if seg_start >= seg_end - new_d:
            ant = min(np.argmax(summed), d - new_d)
        else:
            ant = np.random.randint(seg_start, seg_end - new_d) #深度一维的起始下标在合法且非零的标注层之间随机选取
        '''
        #print('[{}, {}]'.format(seg_start, seg_end - new_d))
        ant = np.random.randint(seg_start, seg_end - new_d) #深度一维的起始下标在合法且非零的标注层之间随机选取
        #print('ant = {} ant + new_d = {} d = {}'.format(ant, ant + new_d, d))
        if self.dataset == 2 or self.dataset == 2.5:
            img_data = img_data[ant: ant + new_d,
                                top: top + new_h, 
                                left: left + new_w]
            img_data = img_data.astype(np.float32) #转为numpy的float32类型
            #新图像尺寸 new_h * new_w * new_d
            
            ant_seg = ant + new_d_half #从图像的深度一维的起始下标加上新深度的一半，对应层的标注作为新图像的标注使用
            seg_data = seg_data[ant_seg: ant_seg + 1,
                                top: top + new_h, 
                                left: left + new_w]
            
            #新标注尺寸 new_h * new_w * 1
            seg_data = seg_data.astype(np.float32) #转为numpy的float32类型


        else:
                
            img_data = img_data[top: top + new_h, 
                                left: left + new_w,
                                ant: ant + new_d]
            img_data = img_data.astype(np.float32) #转为numpy的float32类型
            #新图像尺寸 new_h * new_w * new_d
            
            ant_seg = ant + new_d_half #从图像的深度一维的起始下标加上新深度的一半，对应层的标注作为新图像的标注使用
            seg_data = seg_data[top: top + new_h, 
                                left: left + new_w,
                                ant_seg: ant_seg + 1]
            #新标注尺寸 new_h * new_w * 1
            seg_data = seg_data.astype(np.float32) #转为numpy的float32类型
        '''
        img = img_data.transpose((2, 1, 0))
        print('img.shape = ', img.shape)
        #img = torch.tensor(np.array(img).transpose((2, 1, 0))[0])
        print('img.shape = ', img.shape)
        slice = img[0]
        print('slice.shape = ', slice.shape)
        name = './img/2/{}.png'.format(idx)
        print('name = ', name)
        slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
        slice.save(name)
        '''
        # Merge labels
        if self.dataset != 4:
            seg_data[seg_data > 1] = 1 #标签中所有大于1的部分全部固定为1 
        #导入BTCV数据集时不加上面一行
        
        #print('RandomCrop time = ', time.time() - t1)

        #print('RandomCrop end img.shape = ', img_data.shape)
        return {'image': img_data, 'label': seg_data}


class RandomHorizontalFlip(object): #随机水平翻转
    """Randomly flip the image in the horizontal direction.
    """
    def __call__(self, sample):
        #t1 = time.time()
        if random.uniform(0, 1) < 0.5: #随机生成的0到1的随机数小于0.5时，不做变换
            return sample
        
        # else return flipped sample
        image, label = sample['image'], sample['label']
        #print('RandomHorizontalFlip img.type = {} segmentation.type = {}'.format(type(image), type(label)))
        #print('RandomHF img.shape = ', image.shape)
        #将各行从上到下的顺序颠倒
        image = np.flip(image, axis = 0).copy()
        label = np.flip(label, axis = 0).copy()
        #print('RandomHorizontalFlip time = ', time.time() - t1)
        return {'image': image, 'label': label}

class RandomVerticalFlip(object): #随机垂直翻转
    """Randomly flip the image in the horizontal direction.
    """
    def __call__(self, sample):
        #t1 = time.time()
        if random.uniform(0, 1) < 0.5: #随机生成的0到1的随机数小于0.5时，不做变换
            return sample
        
        # else return flipped sample
        image, label = sample['image'], sample['label']
        #print('RandomVerticalFlip img.type = {} segmentation.type = {}'.format(type(image), type(label)))
        #print('RandomVF img.shape = ', image.shape)
        #将各列从左到右的顺序颠倒
        image = np.flip(image, axis = 1).copy()
        label = np.flip(label, axis = 1).copy()
        #print('RandomVerticalFlip time = ', time.time() - t1)
        return {'image': image, 'label': label}

class Clip(object): #固定数据上下界
    """Clip the intensity values.

    Args:
        Lower and upper bounds.
    """

    def __init__(self, lower_bound, upper_bound):
        '''
        '''
        # Make sure upper bound is larger than the lower bound
        self.LB = min(lower_bound, upper_bound) #数据下界
        self.UB = max(lower_bound, upper_bound) #数据上界

    def __call__(self, sample):
        #t1 = time.time()
        image, label = sample['image'], sample['label']
        #print('Clip img.type = {} segmentation.type = {}'.format(type(image), type(label)))
        #print('Clip img.shape = ', image.shape)
        image[image > self.UB] = self.UB #所有大于上界的部分固定为上界
        image[image < self.LB] = self.LB #所有小于下界的部分固定为下界
        #print('Clip time = ', time.time() - t1)
        return {'image': image, 'label': label}
    
    
class Normalize(object): #图像归一化
    """Normalize the input data to 0 mean 1 std per channel"""
    
    def __init__(self, lower_bound, upper_bound):
        self.LB = min(lower_bound, upper_bound) #数据下界
        self.UB = max(lower_bound, upper_bound) #数据上界

    def __call__(self, sample):
        #t1 = time.time()
        image, label = sample['image'], sample['label']
        #print('Normalize img.type = {} segmentation.type = {}'.format(type(image), type(label)))
        #print('Normalize img.shape = ', image.shape)
        mid_point = (self.LB + self.UB) / 2.0 #上界和下界的均值
        image -= mid_point #图像减去该均值
        half_range = (self.UB - self.LB) / 2.0 #上界到下界范围的一半
        image /= (half_range + 0.000001) #图像除以该范围

        #print('Normalize_image.shape = ', image.shape) 224 * 224 * 3
        #print('Normalize_image.unique = ', np.unique(image)) [-1, 1]
        #print('Normalize time = ', time.time() - t1)
        return {'image': image, 'label': label}
    
    
class ToTensor(object): #将np.array转换成tensor
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __call__(self, sample):
        #t1 = time.time()
        image, label = sample['image'], sample['label']
        #print('ToTensor img.type = {} segmentation.type = {}'.format(type(image), type(label)))
        #print('Tensor img.shape = ', image.shape)
        # swap color axis because
        # numpy image: W x H x C
        # torch image: C X H X W
        
        #array的维度顺序是 012宽高深，而torch的维度顺序是 210深高宽
        #改变array各维度顺序
        if self.dataset != 2 and self.dataset != 2.5:
            image = image.transpose((2, 1, 0))
            label = label.transpose((2, 1, 0))
        #print('ToTensor time = ', time.time() - t1)
        #将array转换成tensor
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}    

def get_composed_transform(hw, slices, view, dataset): #hw = 图片尺寸； slices = 每张图片选取的层数； view = 图像扫描方向
    print('new_shape = {} * {} * {}'.format(hw, hw, slices))
    
    composed = transforms.Compose([RandomCrop((hw, hw, slices), view, dataset), #随机裁剪选取非零标注层
                                   Clip(-200, 200), #固定数据上下界
                                   Normalize(-200, 200), #图像归一化
                                   RandomHorizontalFlip(), #随机水平翻转
                                   RandomVerticalFlip(), #随机竖直翻转
                                   ToTensor(dataset)]) #array转换成tensor
    
    #composed = transforms.Compose([RandomCrop((hw, hw, slices), view, dataset), ToTensor()]) #array转换成tensor
    return composed
    