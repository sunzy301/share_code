# -*-coding:utf-8 -*-
import sys
import os
from multiprocessing import Pool

from skimage import io, transform
from sim_img import calc_sim, get_feature

# 图像文件夹位置
DIR_PATH = './imgs'
# 图像1文件位置
IMG1_FILE_PATH = './imgs/5.jpg'
# 图像2文件位置
IMG2_FILE_PATH = './imgs/6.jpg'
# 默认阈值
THRESHOLD = 7.
# 屏幕输出DEBUG选型
DEBUG = 1
# 多线程处理线程数量
PRCOESS_NUM = 6

multi_process_features = {}

def compare_two_imgs(img1_file_path, img2_file_path, threshold):
    '''
    比较两张图片的相似程度

    Args:
        img1_file_path:图像1文件位置
        img2_file_path:图像2文件位置
        threshold:阈值
    '''
    img1 = io.imread(img1_file_path)
    img2 = io.imread(img2_file_path)
    if DEBUG:
        print('img1 shape ', img1.shape)
        print('img2 shape ', img2.shape)
    r1 = get_feature(img1)
    r2 = get_feature(img2)
    rslt = calc_sim(r1, r2)
    if DEBUG:
        print('colour sim is %f' % rslt[0])
        print('texture sim is %f' % rslt[1])
    sim = rslt[0] + rslt[1]
    is_duplicate = sim > threshold
    return is_duplicate, rslt

def read_img_and_get_feature(img_path):
    '''
    线程的执行函数，包括读入图像和计算颜色与纹理特征直方图。

    Args:
        img_path:图像文件位置
    '''
    tmp_img = io.imread(img_path)
    feature = get_feature(tmp_img)
    return feature

def calc_sim_from_imgs(dir_path, threshold):
    '''
    计算一个文件夹中多张图像之间互相的相似程度，使用多线程处理以加速。

    Args:
        dir_path:图像文件夹位置
        threshold:阈值
    '''
    imgs = []
    # 创建线程池
    p = Pool(PRCOESS_NUM)

    # 遍历目标文件夹下所有图像
    for i, img_file_name in enumerate(os.listdir(dir_path)):
        # 读入图像，并且计算其颜色和特征直方图
        img_path = os.path.join(dir_path, img_file_name)         
        imgs.append(img_file_name)
        multi_process_features[img_file_name] = p.apply_async(
                            read_img_and_get_feature, 
                            (img_path,))

    p.close()
    p.join()

    # 取得多线程实际计算得到的特征值
    features = {}
    for img_name in multi_process_features.keys():
        features[img_name] = multi_process_features[img_name].get()

    sim_matrix = {}
    # 循环比对所有图像之间的相似度，包含图像本身
    for img1_name in imgs:
        sim_matrix[img1_name] = {}
        for img2_name in imgs:
            # 计算两张图像之间的相似度，包含颜色和特征两个维度
            rslt = calc_sim(features[img1_name], features[img2_name])
            sim = rslt[0] + rslt[1]
            sim_matrix[img1_name][img2_name] = rslt

            if DEBUG:
                print('%s %s sim: %f, %f' % (img1_name, img2_name, rslt[0], rslt[1]))

    if DEBUG:
        print(sim_matrix)

    return sim_matrix

def main():
    '''
    主函数，用于测试
    '''
    # is_duplicate, rslt = compare_two_imgs(IMG1_FILE_PATH, IMG2_FILE_PATH, THRESHOLD)
    # sim = rslt[0] + rslt[1]
    # print('The similarity between %s and %s is %f' % 
    #         (IMG1_FILE_PATH, IMG2_FILE_PATH, sim))
    # print('The threshold is %f, and they are %d' % (THRESHOLD, is_duplicate))

    calc_sim_from_imgs(DIR_PATH, THRESHOLD)

if __name__ == '__main__':
    main()
