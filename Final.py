#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 5 16:00:33 2017

@author: yan
"""

# %% train the network
import argparse
import datetime
import math
import numpy as np
import os
from os import path
import shutil
import time

from copy import deepcopy
from torchvision import transforms
from scipy import ndimage
import random
from PIL import Image

import torch
from torch import cuda
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

import dataset.Final_loader as dl
from model.Final_model import ResUNet

parser = argparse.ArgumentParser(description='PyTorch ResUNet Training')
parser.add_argument('--epochs', default=2500, type=int, metavar='N',
                    help='number of total epochs to run') #训练次数
parser.add_argument('-b', '--batchsize', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)') #批次大小
parser.add_argument('--blocksize', default=224, type=int,
                    metavar='N', help='H/W of each image block (default: 224)') #数据尺寸、长宽
parser.add_argument('-s', '--slices', default=3, type=int,
                    metavar='N', help='number of slices (default: 3)') #选取的合法且标注非零的数据层数
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate (default: 0.002)') #学习率
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='N', help='momentum for optimizer (default: 0.9)') #优化器参数更新所用的动量
parser.add_argument('--view', default='axial', type=str,
                    metavar='View', help='view for segmentation (default: axial)') #图像扫描方向 axial = 横轴位 sagittal = 矢状位 coronal = 冠状位

class AverageMeter(object): #用于输出平均值
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self): #初始化
        self.val = 0 #最后一次更新的值
        self.avg = 0 #均值
        self.sum = 0 #总和
        self.count = 0 #数值个数

    def update(self, val, n = 1): #更新
        self.val = val #最后一次更新的值
        self.sum += val * n #总和
        self.count += n #数值个数
        self.avg = self.sum / self.count #计算均值
 
def calc(output, target): #计算某类别的Dice分数
    """Computes the Dice similarity"""
    output = output.float() #预测结果   
    target = target.float() #标注 或 伪标签

    #将[8, 1, 224, 224]的矩阵展开成[8, 50176]的向量，便于计算交和并
    seg_channel = output.view(output.size(0), -1) #展开预测结果
    target_channel = target.view(target.size(0), -1) #展开标注
    
    #print('output.shape = {} seg_channel = {}'.format(output.shape, seg_channel.shape))
    #[8, 1, 224, 224] -> [8, 50176]
    #print('target.shape = {} target_channel = {}'.format(target.shape, target_channel.shape))
    #[8, 224, 224] -> [8, 50176]
    
    intersection = (seg_channel * target_channel).sum()  #交
    union = (seg_channel + target_channel).sum() #并

    smooth = 0.00001 #加到分母上，防止除0
    dice = (2. * intersection) / (union + smooth) # Dice = 2 * 交 / 并

    return torch.mean(dice) #返回批次的Dice分数的均值
        
def dice_similarity(output, target): #计算总的Dice分数
    """Computes the Dice similarity"""
    total_dice = 0 #初始化总Dice分数
    
    output = output.clone() #预测结果
    target = target.clone() #标注

    #output.shape = [3, 2, 224, 224] target.shape = [3, 224, 224]
    for i in range(1, output.shape[1]): #枚举所有类别（没有背景），计算对应类别的预测结果与标注的Dice分数
        target_i = torch.zeros(target.shape)
        target_i = target_i.cuda().clone() #初始化为和标注一样大的全零矩阵
        target_i[target == i] = 1 #只将属于当前类别的部分标为1
        
        output_i = output[:, i:i+1].clone() #取出当前类别对应的预测结果

        dice_i = calc(output_i, target_i) #计算当前类别对应的预测结果与标注的Dice分数

        total_dice += dice_i #加到总Dice分数中
    total_dice = total_dice / (output.shape[1] - 1) #计算平均Dice分数，不包括背景所以分母要减1

    return total_dice 

def obtain_cutmix_box(img_size, size_min = 0.02, size_max = 0.4, ratio_1 = 0.3, ratio_2 = 1 / 0.3): #获取CutMix所用的混合区域
    mask = torch.zeros(img_size, img_size) #用于表明混合区域，初始化为和原图一样大的全零矩阵
    size = np.random.uniform(size_min, size_max) * img_size * img_size #随机得到一个size_min与size_max之间的比例，进而计算出混合区域的大小
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2) #随机生成一个比例系数
        cutmix_w = int(np.sqrt(size / ratio)) #计算混合区域的宽
        cutmix_h = int(np.sqrt(size * ratio)) #计算混合区域的高
        
        #随机选取混合区域的左上角坐标
        x = np.random.randint(0, img_size) 
        y = np.random.randint(0, img_size)

        #判断混合区域是否会超出原图边界
        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1 #将混合区域的部分设为1
    return mask

def merge(pred, task): #合并预测结果
    C = pred.shape[1] #类别个数
    pred_p2 = pred[:,task:task+1,:,:].clone() #p2 = 所属数据集对应类别
    pred_p1 = pred[:,0:1,:,:].clone() 
    for i in range(1, C):
        pred_p1 += pred[:,i:i+1,:,:].clone()
    pred_p1 -= pred_p2 #p1 = 除p2外的所有预测结果之和
    pred_p = torch.cat((pred_p1, pred_p2), 1) #合并两个预测结果
    return pred_p

#半监督训练
def unsup_train(labeled_loader, unlabeled_loader, data_type, model, criterion, optimizer, epoch, verbose=True):
    """Function for training"""
    batch_time = AverageMeter() #批次训练用时
    total_loss = AverageMeter() #总损失函数值
    dice = AverageMeter() #Dice分数

    # switch to train mode
    model.train() #模型转为训练模式

    end_time = time.time() #初始化训练结束时间

    train_loader = zip(labeled_loader, unlabeled_loader, unlabeled_loader) #组合三类数据集导入器（已标注数据，未标注数据，未标注数据（用于CutMix））
    
    for i, (labeled_sample, unlabeled_sample, mix_sample) in enumerate(train_loader): #从训练数据集中提取数据

        #已标注数据 用于监督训练
        img_x = Variable(labeled_sample['image']).float().cuda() #监督训练用图像
        label_x = Variable(labeled_sample['label'][:,0,:,:]).long().cuda() #监督训练用标签
        '''
        N = img_x.shape[0]
        for j in range(N):
            slice = np.array(img_x[j:j+1, 0:1, :, :].squeeze(dim = 0).squeeze(dim = 0).cpu())
            #print('slice.shape = ', slice.shape)
            name = './img/x/{}-{}-{}.png'.format(epoch, i, j)
            #print('name = ', name)
            slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
            slice.save(name)

        for j in range(N):
            slice = np.array(label_x[j:j+1][0].cpu())
            #print('slice.shape = ', slice.shape)
            name = './img/xl/{}-{}-{}.png'.format(epoch, i, j)
            #print('name = ', name)
            slice = Image.fromarray((slice * 255).astype('uint8')).convert('L')
            slice.save(name)
        '''
        
        #未标注数据 用于半监督训练
        img_w = Variable(unlabeled_sample['image']).float().cuda() #弱扰动图像
        img_s1 = Variable(unlabeled_sample['image_s1']).float().cuda() #强扰动图像1
        img_s2 = Variable(unlabeled_sample['image_s2']).float().cuda() #强扰动图像2

        #未标注数据 用于CutMix
        img_w_mix = Variable(mix_sample['image']).float().cuda() #弱扰动图像的mix
        img_s1_mix = Variable(mix_sample['image_s1']).float().cuda() #强扰动图像1的mix
        img_s2_mix = Variable(mix_sample['image_s2']).float().cuda() #强扰动图像2的mix

        #获取两个强扰动图像的CutMix混合区域
        cutmix_s1 = obtain_cutmix_box(img_w.shape[2])
        cutmix_s2 = obtain_cutmix_box(img_w.shape[2])

        
        #对于半监督训练所用的强扰动图像，将其混合区域的部分用对应mix图像中的混合区域替换
        img_s1[cutmix_s1.unsqueeze(0).unsqueeze(1).expand(img_s1.shape) == 1] = img_s1_mix[cutmix_s1.unsqueeze(0).unsqueeze(1).expand(img_s1.shape) == 1]
        img_s2[cutmix_s2.unsqueeze(0).unsqueeze(1).expand(img_s2.shape) == 1] = img_s2_mix[cutmix_s2.unsqueeze(0).unsqueeze(1).expand(img_s2.shape) == 1]
        
        
        '''
        N = img_w.shape[0]
        for j in range(N):
            slice = np.array(img_w[j:j+1, 0:1, :, :].squeeze(dim = 0).squeeze(dim = 0).cpu())
            #print('slice.shape = ', slice.shape)
            name = './input_test/LiTS-w/img_w-{}-{}.png'.format(i, j)
            #print('name = ', name)
            slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
            slice.save(name)

        for j in range(N):
            slice = np.array(img_s1[j:j+1, 0:1, :, :].squeeze(dim = 0).squeeze(dim = 0).cpu())
            #print('slice.shape = ', slice.shape)
            name = './input_test/LiTS-s1/img_s1-{}-{}.png'.format(i, j)
            #print('name = ', name)
            slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
            slice.save(name)
        '''
        
        #各img都是[N = 8, 3, 224, 224]
        #print('img_x.dtype = {} img_w.dtype = {} img_s1.dtype = {} img_s2.dtype = {}'.format(img_x.dtype, img_w.dtype, img_s1.dtype, img_s2.dtype))
        #计算mix数据的置信度和伪标签
        with torch.no_grad(): #内部数据不计算梯度，不参与反向传播
            model.eval() #将模型转化为评估模式
            pred_w_mix, _ = model(img_w_mix) #计算mix数据的预测结果
            pred_w_mix = pred_w_mix.clamp_min_(1e-10) #防止有零存在
            pred_w_mix = pred_w_mix.detach() #令mix数据预测结果不参与反向传播
            conf_w_mix = pred_w_mix.softmax(dim = 1).max(dim = 1)[0] #计算mix数据的置信度
            label_w_mix = pred_w_mix.argmax(dim = 1) #计算mix数据的伪标签
        
        model.train() #模型转换为训练模式
        pred_x, _ = model(img_x) #输入已标注数据，计算已标注数据的预测结果
        pred_w, pred_fp = model(img_w) #输入弱扰动的未标注数据，计算弱扰动数据、特征级扰动数据的预测结果
        pred_s1, _ = model(img_s1) #输入强扰动的未标注数据1，计算强扰动数据1的预测结果
        pred_s2, _ = model(img_s2) #输入强扰动的未标注数据2，计算强扰动数据2的预测结果
        #print('img_x.shape = {} pred_x.shape = {}'.format(img_x.shape, pred_x.shape))
        pred_x_p = merge(pred_x, data_type).clamp_min_(1e-10) #合并已标注数据预测结果
        pred_w_p = merge(pred_w, data_type).clamp_min_(1e-10) #合并弱扰动数据预测结果
        pred_fp_p = merge(pred_fp, data_type).clamp_min_(1e-10) #合并特征级扰动数据预测结果
        pred_s1_p = merge(pred_s1, data_type).clamp_min_(1e-10) #合并强扰动数据1预测结果
        pred_s2_p = merge(pred_s2, data_type).clamp_min_(1e-10) #合并强扰动数据2预测结果

        pred_w_p = pred_w_p.detach() #令弱扰动数据预测结果不参与反向传播
        conf_w = pred_w_p.max(dim = 1)[0] #取各类别预测结果的最大值作为置信度
        label_w = pred_w_p.argmax(dim = 1) #取各类别预测结果最大值对应的类别作为伪标签
        
        conf_s1 = conf_w #初始化强扰动数据1的置信度为弱扰动数据的置信度
        conf_s1[cutmix_s1.unsqueeze(0).expand(conf_w.shape) == 1] = conf_w_mix[cutmix_s1.unsqueeze(0).expand(conf_w.shape) == 1] 
        #替换强扰动数据1的置信度的混合区域为mix的置信度的混合区域
        conf_s2 = conf_w #初始化强扰动数据2的置信度为弱扰动数据的置信度
        conf_s2[cutmix_s2.unsqueeze(0).expand(conf_w.shape) == 1] = conf_w_mix[cutmix_s2.unsqueeze(0).expand(conf_w.shape) == 1]
        #替换强扰动数据2的置信度的混合区域为mix的置信度的混合区域
        
        label_s1 = label_w #初始化强扰动数据1的伪标签为弱扰动数据的伪标签
        label_s1[cutmix_s1.unsqueeze(0).expand(label_w.shape) == 1] = label_w_mix[cutmix_s1.unsqueeze(0).expand(label_w.shape) == 1]
        #替换强扰动数据1的伪标签的混合区域为mix的伪标签的混合区域
        label_s2 = label_w #初始化强扰动数据2的伪标签为弱扰动数据的伪标签
        label_s2[cutmix_s2.unsqueeze(0).expand(label_w.shape) == 1] = label_w_mix[cutmix_s2.unsqueeze(0).expand(label_w.shape) == 1]
        #替换强扰动数据2的伪标签的混合区域为mix的伪标签的混合区域
        #print('label_w.shape = {} label_s1.shape = {}'.format(label_w.shape, label_s1.shape)) #[N = 8, 224, 224]
        print('label_w.unique = {} label_s1.unique = {}'.format(torch.unique(label_w), torch.unique(label_s1))) 
        '''
        N = label_w.shape[0]
        for j in range(N):
            slice = np.array(label_w[j:j+1][0].cpu())
            print('slice.shape = ', slice.shape)
            name = './image_test/Tr-label_w/Tr-label_w-{}-{}.png'.format(i, j)
            print('name = ', name)
            slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
            slice.save(name)
        '''
        '''
        for j in range(N):
            slice = np.array(label_s1[j:j+1][0].cpu())
            print('slice.shape = ', slice.shape)
            name = './image_test/Tr-label_s1/Tr-label_s1-{}-{}.png'.format(i, j)
            print('name = ', name)
            slice = Image.fromarray((slice * 255).astype('uint8')).convert('L')
            slice.save(name)
        '''
        #置信阈值，超过此阈值的伪标签视为可信标签
        if epoch <= 700:
            threshold = 0.7
        elif epoch <= 1400:
            threshold = 0.8
        else:
            threshold = 0.9
        
        loss_x = criterion[0](torch.log(pred_x_p), label_x) #计算已标注数据的损失函数
        
        loss_fp = criterion[1](torch.log(pred_fp_p), label_w) #计算特征级扰动数据的损失函数
        loss_fp = loss_fp * (conf_w >= threshold) #取其中置信度大于阈值的部分
        loss_fp = loss_fp.sum() / (label_w != 255).sum().item() #取平均
        
        loss_s1 = criterion[1](torch.log(pred_s1_p), label_s1) #计算强扰动数据1的损失函数
        loss_s1 = loss_s1 * (conf_s1 >= threshold)
        loss_s1 = loss_s1.sum() / (label_s1 != 255).sum().item()
        
        loss_s2 = criterion[1](torch.log(pred_s2_p), label_s2) #计算强扰动数据2的损失函数
        loss_s2 = loss_s2 * (conf_s2 >= threshold)
        loss_s2 = loss_s2.sum() / (label_s2 != 255).sum().item()

        loss_u = loss_fp * 0.5 + loss_s1 * 0.25 + loss_s2 * 0.25 #计算半监督损失函数值
        
        loss = (loss_x + loss_u) / 2.0 #计算最终损失函数值
        print('loss_x = {} loss_u = {}'.format(loss_x, loss_u))
        
        total_loss.update(loss.data, img_x.size(0)) #更新累积的损失函数值与均值等等

        ds_x = dice_similarity(pred_x_p, label_x) #计算已标注图像预测结果的Dice分数
        
        ds_fp = dice_similarity(pred_fp_p, label_w) #计算特征级扰动预测结果的Dice分数
        ds_s1 = dice_similarity(pred_s1_p, label_s1) #计算强扰动图像1预测结果的Dice分数
        ds_s2 = dice_similarity(pred_s2_p, label_s2) #计算强扰动图像2预测结果的Dice分数
        ds_u = (ds_fp.data + ds_s1.data + ds_s2.data) / 3.0 #计算三者均值作为未标注图像预测结果的Dice分数
    
        ds_tr = (ds_x.data + ds_u.data) / 2.0 #计算已标注图像Dice分数和未标注图像Dice分数的均值，作为本轮训练的Dice分数

        print('dice_tr = {} dice_x = {} dice_u = {}'.format(ds_tr, ds_x.data, ds_u))
        print('dice_fp = {} dice_s1 = {} dice_s2 = {}'.format(ds_fp.data, ds_s1.data, ds_s2.data))
        
        dice.update(ds_tr, img_x.size(0)) #更新累计的Dice分数与均值等等
        
        optimizer.zero_grad() #初始化清零梯度
        loss.backward() #损失函数反向传播
        optimizer.step() #优化器更新参数

        batch_time.update(time.time() - end_time) #更新累计批次用时与均值等等
        end_time = time.time() #更新结束时间

        if ((i + 1) % 10 == 0) and verbose: #每10批次训练输出一次信息（用时、损失函数值、Dice分数的均值）
            print('Train ep {0} [batch {1}/{2}]: Time {batch_time.val:.1f}s, Loss avg: {loss.avg:.4f}, Dice avg: {dice.avg:.4f}'.format(
                epoch + 1, i + 1, len(list(train_loader)), batch_time = batch_time, loss = total_loss, dice = dice), flush = True)

    #本次训练结束后，输出损失函数值的均值和Dice分数的均值
    if data_type == 1:
        dataset_name = 'LiTS'
    elif data_type == 2:
        dataset_name = 'KiTS'
    elif data_type == 3:
        dataset_name = 'MSDSpleen'
    print('{} Training -> Loss: {loss.avg:.4f}, Dice {dice.avg:.3f}'.format(dataset_name, loss = total_loss, dice = dice), flush = True)

    return total_loss.avg, dice.avg

#验证
def validate(loader, data_type, model, criterion, epoch, verbose = True):
    batch_time = AverageMeter() #当前批次验证用时
    losses = AverageMeter() #损失函数值
    dice = AverageMeter() #Dice分数

    # switch to evaluate mode
    model.eval() #将模型转为评估模式

    end = time.time() #初始化结束时间为当前时间
    for i, sample_batched in enumerate(loader): #从验证集中提取数据
        
        img = Variable(sample_batched['image'], volatile = True).float().cuda() #验证集图像 [8, 3, 224, 224]
        label = Variable(sample_batched['label'][:, 0, :, :], volatile = True).long().cuda() #验证集标注 [8, 1, 224, 224] -> [8, 224, 224]
        #volatile = True表示反向传播时不会自动求导

        '''
        N = label.shape[0]
        for j in range(N):
            slice = np.array(img[j:j+1, 0:1, :, :].squeeze(dim = 0).squeeze(dim = 0).cpu())
            print('slice.shape = ', slice.shape)
            name = './img/v/{}-{}-{}.png'.format(epoch, i, j)
            print('name = ', name)
            slice = Image.fromarray((slice * 255).astype('float32')).convert('L')
            slice.save(name)
        
        for j in range(N):
            slice = np.array(label[j:j+1][0].cpu())
            print('slice.shape = ', slice.shape)
            name = './img/vl/{}-{}-{}.png'.format(epoch, i, j)
            print('name = ', name)
            slice = Image.fromarray((slice * 255).astype('uint8')).convert('L')
            slice.save(name)
        '''
        
        pred, _ = model(img) #获取验证集预测结果

        #合并验证集预测结果
        if data_type == 4: #4对应全标注数据集BTCV，不需要合并预测结果
            pred_p = pred.clone()
        else: #其他数字对应不同的部分标注数据集，需要将不属于对应数据集类别的预测结果合并
            C = pred.shape[1]
            pred_p = pred[:, data_type - 1:data_type + 1, :, :].clone() #合并结果取[类别-1, 类别]的两部分，合并结果的0对应背景，1对应数据集所用类别
            pred_p[:, 0, :, :] = pred[:, 0, :, :].clone()
            for i in range(1, C):
                pred_p[:, 0, :, :] += pred[:, i, :, :].clone()
            pred_p[:, 0, :, :] -= pred[:, data_type, :, :].clone() #将非数据集对应类别的预测结果相加，作为0对应的背景的预测结果

        pred_p = pred_p.clamp_min_(1e-10)
    
        loss = criterion[0](torch.log(pred_p), label) #计算验证集损失函数值
        losses.update(loss.data, img.size(0)) #更新累积的损失函数值与均值等等

        ds = dice_similarity(pred_p, label) #计算验证集的Dice分数
        dice.update(ds.data, img.size(0)) #更新累计的Dice分数与均值等等

        batch_time.update(time.time() - end) #更新累计批次用时与均值等等
        end = time.time() #更新结束时间

        if ((i + 1) % 10 == 0) and verbose: #每10批次验证输出一次信息（用时、损失函数值、Dice分数的均值）
            print('Validation ep {0} [batch {1}/{2}]: Time {batch_time.val:.1f}s, Loss avg: {loss.avg:.4f}, Dice avg: {dice.avg:.4f}'.format(
                epoch + 1, i + 1, len(list(loader)), batch_time = batch_time, loss = losses, dice = dice), flush = True)

    #本次验证结束后，输出损失函数值的均值和Dice分数的均值
    if data_type == 1:
        dataset_name = 'LiTS'
    elif data_type == 2:
        dataset_name = 'KiTS'
    elif data_type == 3:
        dataset_name = 'MSDSpleen'
    print('{} Validation ep {} -> loss: {loss.avg:.4f}, Dice {dice.avg:.3f}'.format(dataset_name, epoch + 1, loss = losses, dice = dice), flush = True)

    return losses.avg, dice.avg


#def adjust_learning_rate(optimizer, epoch):
def adjust_learning_rate(optimizer, gamma = 0.9): #调整学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] *= gamma #所有学习率乘上一个衰减系数


        
#保存模型参数checkpoint
def save_checkpoint(state, is_best, log_folder, view = 'axial', filename = 'checkpoint.pth.tar'):
    """Save checkpoints
    """
    filename = path.join(log_folder, filename) #存放checkpoint文件的路径
    torch.save(state, filename) #保存模型参数
    if is_best: #如果是当前性能最佳模型，则拷贝一份作为当前最佳模型参数
        filename_best = path.join(log_folder, 'resu_best_{}.pth.tar'.format(view))
        shutil.copyfile(filename, filename_best)

if __name__ == "__main__":
    
    global args
    args = parser.parse_args() #参数集
    cv = args.cv_n #（没用到）
    use_cuda = cuda.is_available() #显卡可用时使用cuda
    
    checkpoing_dir = path.expanduser('./ckp-LiTS-semi') #checkpoint保存路径
    if not path.isdir(checkpoing_dir): 
        os.makedirs(checkpoing_dir)
        
    log_dir = path.expanduser('./log-LiTS-semi') #日志保存路径
    if not path.isdir(log_dir):
        os.makedirs(log_dir)

    num_classes = 4 #数据类别的个数
    num_in_channels = args.slices #选取的合法且标注非零的数据层数

    model = ResUNet(num_in_channels, num_classes) #模型
    '''
    resunet_checkpoint = torch.load('./ckp-LiTS/resu_best_axial.pth.tar')
    resunet_dict = resunet_checkpoint['state_dict']

    model.resnet.load_state_dict(resunet_dict)
    '''
    optimizer = optim.RMSprop(model.parameters(), lr = args.lr, momentum = args.momentum) #优化器

    #数据集1路径
    folder_labeled_1 = './data/LiTS/labeled' #已标注数据路径
    folder_unlabeled_1 = './data/LiTS/unlabeled' #未标注数据路径
    folder_validation_1 = './data/LiTS/validation' #验证集路径
        
    #数据集2路径
    folder_labeled_2 = './data/KiTS/new_labeled' #已标注数据路径
    folder_unlabeled_2 = './data/KiTS/new_unlabeled' #未标注数据路径
    folder_validation_2 = './data/KiTS/new_validation' #验证集路径

    #数据集3路径
    folder_labeled_3 = './data/MSDSpleen/labeled' #已标注数据路径
    folder_unlabeled_3 = './data/MSDSpleen/unlabeled' #未标注数据路径
    folder_validation_3 = './data/MSDSpleen/validation' #验证集路径

    #数据集4路径
    folder_labeled_4 = './data/BTCV/labeled' #已标注数据路径
    folder_unlabeled_4 = './data/BTCV/unlabeled' #未标注数据路径
    folder_validation_4 = './data/BTCV/validation' #验证集路径

    weights = torch.Tensor([0.2, 1.2])
    criterion_l = nn.NLLLoss2d(weight = weights)#计算监督损失和验证集损失时使用
    criterion_u = nn.NLLLoss2d(reduction = 'none', weight = weights) #计算半监督损失时使用，不取平均，直接以矩阵形式输出
    criterion = [criterion_l, criterion_u] #损失函数
    
    if use_cuda: #使用CUDA
        print('\n***** Training ResU-Net with GPU *****\n')
        model.cuda()
        criterion[0].cuda()
        criterion[1].cuda()

    blocksize = args.blocksize #数据尺寸
    view = args.view #图像扫描方向
    if view != 'axial' and view != 'sagittal' and view != 'coronal': 
        print('The given view of <{}> is not supported!'.format(view))
  
    batchsize = args.batchsize #批次大小

    #数据集1
    composed = dl.get_composed_transform(blocksize, num_in_channels, view, 1) #初始化图像变换组合
    #已标注数据集1
    dataset_labeled1 = dl.PartialDataset(folder_labeled_1, transform = composed, type = 'nii')
    labeled_loader1 = dl.DataLoader(dataset_labeled1, batch_size = args.batchsize, shuffle = True, num_workers = 0, drop_last = False)
    #验证集1
    dataset_validation1 = dl.PartialDataset(folder_validation_1, transform = composed, type = 'nii')
    val_loader1 = dl.DataLoader(dataset_validation1, batch_size = args.batchsize, shuffle = False, num_workers = 0, drop_last = False)
    composed = dl.get_composed_transform(blocksize, num_in_channels, view, 1.5) #初始化图像变换组合
    #未标注数据集1
    dataset_unlabeled1 = dl.PartialDataset(folder_unlabeled_1, transform = composed, type = 'nii')
    unlabeled_loader1 = dl.DataLoader(dataset_unlabeled1, batch_size = args.batchsize, shuffle = True, num_workers = 0, drop_last = False)
    
    
    #数据集2
    composed = dl.get_composed_transform(blocksize, num_in_channels, view, 2)
    #已标注数据集2
    dataset_labeled2 = dl.PartialDataset(folder_labeled_2, transform = composed, type = 'nii.gz')
    labeled_loader2 = dl.DataLoader(dataset_labeled2, batch_size = args.batchsize, shuffle = True, num_workers = 0, drop_last = False)
    #验证集2
    dataset_validation2 = dl.PartialDataset(folder_validation_2, transform = composed, type = 'nii.gz')
    val_loader2 = dl.DataLoader(dataset_validation2, batch_size = args.batchsize, shuffle = False, num_workers = 0, drop_last = False)
    composed = dl.get_composed_transform(blocksize, num_in_channels, view, 2.5)
    #未标注数据集2
    dataset_unlabeled2 = dl.PartialDataset(folder_unlabeled_2, transform = composed, type = 'nii.gz')
    unlabeled_loader2 = dl.DataLoader(dataset_unlabeled2, batch_size = args.batchsize, shuffle = True, num_workers = 0, drop_last = False)
    

    #数据集3
    composed = dl.get_composed_transform(blocksize, num_in_channels, view, 3)
    #已标注数据集3
    dataset_labeled3 = dl.PartialDataset(folder_labeled_3, transform = composed, type = 'nii.gz')
    labeled_loader3 = dl.DataLoader(dataset_labeled3, batch_size = args.batchsize, shuffle = True, num_workers = 0, drop_last = False)
    #验证集3
    dataset_validation3 = dl.PartialDataset(folder_validation_3, transform = composed, type = 'nii.gz')
    val_loader3 = dl.DataLoader(dataset_validation3, batch_size = args.batchsize, shuffle = False, num_workers = 0, drop_last = False)
    composed = dl.get_composed_transform(blocksize, num_in_channels, view, 3.5)
    #未标注数据集3
    dataset_unlabeled3 = dl.PartialDataset(folder_unlabeled_3, transform = composed, type = 'nii.gz')
    unlabeled_loader3 = dl.DataLoader(dataset_unlabeled3, batch_size = args.batchsize, shuffle = True, num_workers = 0, drop_last = False)
    
    
    #数据集4
    composed = dl.get_composed_transform(blocksize, num_in_channels, view, 4)
    #已标注数据集4
    dataset_labeled4 = dl.PartialDataset(folder_labeled_4, transform = composed, type = 'nii.gz')
    labeled_loader4 = dl.DataLoader(dataset_labeled4, batch_size = args.batchsize, shuffle = True, num_workers = 0, drop_last = False)
    #验证集4
    dataset_validation4 = dl.PartialDataset(folder_validation_4, transform = composed, type = 'nii.gz')
    val_loader4 = dl.DataLoader(dataset_validation4, batch_size = args.batchsize, shuffle = False, num_workers = 0, drop_last = False)
    #composed = dl.get_composed_transform(blocksize, num_in_channels, view, 4.5)
    #未标注数据集4
    dataset_unlabeled4 = dl.PartialDataset(folder_unlabeled_4, transform = composed, type = 'nii.gz')
    unlabeled_loader4 = dl.DataLoader(dataset_unlabeled4, batch_size = args.batchsize, shuffle = True, num_workers = 0, drop_last = False)
    
    
    best_dice = -1.0 #初始化最佳Dice分数

    num_epochs = args.epochs #训练次数
    
    train_history = [] #每次训练的损失函数值
    val_history = [] #每次验证的损失函数值

    #半监督训练
    for epoch in range(num_epochs):
        print('Unsup Training epoch {} of {}...'.format(epoch + 1, num_epochs))
        # start timing
        t_start = time.time() #训练开始时间
        
        #半监督训练
        if epoch % 3 == 0:
            unsup_train_loss = unsup_train(labeled_loader1, unlabeled_loader1, 1, model, criterion, optimizer, epoch, verbose = True)
        elif epoch % 3 == 1:
            unsup_train_loss = unsup_train(labeled_loader2, unlabeled_loader2, 2, model, criterion, optimizer, epoch, verbose = True)
        elif epoch % 3 == 2:
            unsup_train_loss = unsup_train(labeled_loader3, unlabeled_loader3, 3, model, criterion, optimizer, epoch, verbose = True)
      
        train_history.append(unsup_train_loss) #保存本次训练的损失函数值
        
        if epoch % 40 == 0: #每训练40轮更新学习率
            adjust_learning_rate(optimizer, gamma=0.99)
        
        #验证
        if epoch % 3 == 0:
            val_loss = validate(val_loader1, 1, model, criterion, epoch, verbose = True)
        elif epoch % 3 == 1:
            val_loss = validate(val_loader2, 2, model, criterion, epoch, verbose = True)
        elif epoch % 3 == 2:
            val_loss = validate(val_loader3, 3, model, criterion, epoch, verbose = True)
        
        val_history.append(val_loss) #保存本次验证的损失函数值

        elapsed_time = time.time() - t_start #训练 + 验证结束时间
        print('Epoch {} completed in {:.2f}s\n'.format(epoch + 1, elapsed_time)) #输出本轮训练 + 验证用时

        dice = val_loss[1] #本次验证的Dice分数
        is_best = dice > best_dice #判断是否为最佳Dice分数
        best_dice = max(dice, best_dice) #更新最佳Dice分数

        if is_best: #如果是最佳Dice分数，则保存模型参数
            fn_checkpoint = 'resu_checkpoint_ep{:04d}.pth.tar'.format(epoch + 1) #模型参数保存路径
            save_checkpoint({'epoch': epoch + 1, #训练轮数
                             'state_dict': model.state_dict(), #模型参数
                             'best_dice': best_dice, #最佳Dice分数
                             'optimizer' : optimizer.state_dict(),}, #优化器参数
                            is_best, #是否为最佳Dice分数
                            checkpoing_dir, #checkpoint保存路径
                            view, #图像扫描方向
                            filename = fn_checkpoint) #模型参数保存路径
            
        if epoch == num_epochs - 1: #若为最后一次训练，则保存模型参数
            filename = path.join(checkpoing_dir, 'resunet_checkpoint_final.pth.tar') #模型参数保存路径
            torch.save({'epoch': epoch + 1, #训练轮数
                        'state_dict': model.state_dict(), #模型参数
                        'best_dice': best_dice, #是否为最佳Dice分数
                        'optimizer' : optimizer.state_dict(),}, #优化器参数
                        filename) #模型参数保存路径
            
    # save the training history
    time_now = datetime.datetime.now() #当前时间
    time_str = time_now.strftime('%y%m%d-%H%M%S')

    fn_train_history = path.join(log_dir, 'train_hist_{}.npy'.format(time_str)) #每次训练的损失函数值的保存路径
    fn_val_history = path.join(log_dir, 'val_hist_{}.npy'.format(time_str)) #每次验证的损失函数值的保存路径
    
    np.save(fn_train_history, np.asarray(train_history)) #保存每次训练的损失函数值
    np.save(fn_val_history, np.asarray(val_history)) #保存每次验证的损失函数值

    #输出训练结束时间及损失函数值保存的位置
    time_disp_str = time_now.strftime('%H:%M:%S on %Y-%m-%d')
    print('Training completed at {}'.format(time_disp_str))
    print('Training history saved into:\n<{}>'.format(fn_train_history))
    print('<{}>'.format(fn_val_history))
