# -*-coding:utf-8 -*-
import os
import math
import shutil

import caffe
import numpy as np

# 图像文件夹位置
DIR_PATH = './imgs/imgs'
# 图像文件位置
IMG_PATH = './imgs/1.jpg'
# 输出文件夹
OUTPUT_DIR = './imgs/output'
# 每张图像输出相似图像数目
OUTPUT_NUM = 5

# VGG下载地址 
# https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
# caffe prototxt文件位置
PROTOTXT_PATH = './models/vgg19.prototxt'
# caffemodel文件位置
CAFFEMODEL_PATH = './models/VGG_ILSVRC_19_layers.caffemodel'

DEBUG = 1

def calc_sim_euc(feature1, feature2):
    '''
    计算两个特征向量之间欧氏距离
    '''
    assert feature1.shape[0] == feature2.shape[0]
    array_len = feature1.shape[0]
    array_sum = 0
    for i in range(array_len):
        array_sum += (feature1[i] - feature2[i]) * \
                        (feature1[i] - feature2[i])
    # print(array_sum)
    return math.sqrt(array_sum)

def calc_sim_cos(feature1, feature2):
    '''
    计算两个特征向量之间cos夹角
    '''
    assert feature1.shape[0] == feature2.shape[0]
    array_len = feature1.shape[0]
    sum1 = 0
    sum2 = 0
    sumall = 0
    for i in range(array_len):
        sum1 += feature1[i] * feature1[i]
        sum2 += feature2[i] * feature2[i]
        sumall += feature1[i] * feature2[i]
    return sumall / (math.sqrt(sum1) * math.sqrt(sum2) + 1e-6)


def extract_feature(img_path):
    '''
    提取单张图像特征

    Args:
        img_path：图像文件位置
    '''
    caffe.set_mode_gpu()
    net = caffe.Net(PROTOTXT_PATH, CAFFEMODEL_PATH, caffe.TEST)
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load('./models/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR 

    image = caffe.io.load_image(img_path)
    transformed_image = transformer.preprocess('data', image)

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()

    # for each layer, show the output shape
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)
    feature = net.blobs['fc7'].data.squeeze()
    print(feature.shape, feature)

def extract_feature_from_dir(dir_path, output_dir):
    '''
    计算一个文件夹中多张图像之间互相的相似程度

    Args:
        dir_path:图像文件夹位置
        output_dir:结果输出文件夹位置
    '''
    # 设置caffe运行和预处理参数
    caffe.set_mode_gpu()
    net = caffe.Net(PROTOTXT_PATH, CAFFEMODEL_PATH, caffe.TEST)
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load('./models/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR 

    features = {}
    imgs = []
    # 读入文件夹下每张图像，并且使用CNN提取特征
    for i, img_file_name in enumerate(os.listdir(dir_path)):
        imgs.append(img_file_name)
        img_path = os.path.join(dir_path, img_file_name)
        image = caffe.io.load_image(img_path)
        transformed_image = transformer.preprocess('data', image)
        # print(transformed_image)
        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image

        ### perform classification
        output = net.forward()

        feature = net.blobs['fc7'].data.copy()
        feature = feature.squeeze()

        features[img_file_name] = feature
        print(i, img_file_name, len(features[img_file_name]))

    sim_matrix = {}
    # 循环比对所有图像之间的相似度，包含图像本身
    for img1_name in imgs:
        sim_matrix[img1_name] = {}
        for img2_name in imgs:
            # 计算两张图像之间的相似度
            rslt = calc_sim_euc(features[img1_name], features[img2_name])
            sim_matrix[img1_name][img2_name] = rslt

            if DEBUG:
                print('%s %s sim: %f' % (img1_name, img2_name, rslt))

    # if DEBUG:
    #     print(sim_matrix)
    imgs.sort()
    for img_name in imgs:
        img_short_name = img_name[:img_name.rfind('.')]
        img_output_dir = os.path.join(output_dir, img_short_name)
        if os.path.exists(img_output_dir):
            os.system('rm -rf %s' % img_output_dir)
        os.makedirs(img_output_dir)

        # 基于相似度进行排序
        temp_dict = sim_matrix[img_name]
        sorted_dict = sorted(temp_dict.items(), key=lambda d: d[1])
        if DEBUG:
            print(temp_dict)
            print(sorted_dict)

        output_index = 0
        for key, value in sorted_dict:
            # 拷贝原图
            if key == img_name:
                dist_img_path = os.path.join(img_output_dir, key)
                raw_img_path = os.path.join(dir_path, key)
                shutil.copy(raw_img_path, dist_img_path)
                continue

            # 输出数量达到要求
            if output_index == OUTPUT_NUM:
                break
            
            output_index += 1
            dist_img_short_name = key[:key.rfind('.')]
            dist_img_name = '%d_%s' % (output_index, key)
            dist_img_path = os.path.join(img_output_dir, dist_img_name)
            raw_img_path = os.path.join(dir_path, key)
            if DEBUG:
                print(raw_img_path, dist_img_path)

            shutil.copy(raw_img_path, dist_img_path)


    if DEBUG:
        print('=============SIM MATRIX================')
        for img1_name in imgs:
            temp_str = 'img %s sim:' % img1_name
            for img2_name in imgs:
                sim = sim_matrix[img1_name][img2_name]
                temp_str += '%.3f   ' % sim
            print(temp_str)
        # print(sim_matrix)

def main():
    '''
    测试函数
    '''
    # extract_feature(IMG_PATH)
    extract_feature_from_dir(DIR_PATH, OUTPUT_DIR)

if __name__ == '__main__':
    main()
