# -*- coding: utf-8 -*-
# /*
#  * @Author: Zhongyi Sun s00425426 
#  * @Date: 2018-05-07 09:35:04 
#  * @Last Modified by:   Zhongyi Sun s00425426 
#  * @Last Modified time: 2018-05-07 09:35:04 
#  */
import os
import sys
import pickle
import json

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT_DIR)

import numpy as np
import lmdb
import caffe

from util.predict_result import PredictResult

def lmdbkey_to_id(key):
    return int(key.split('_')[0])

def predict(prototxt_path,
            caffemodel_path,
            lmdb_path, 
            cur_json_file, 
            gpu_num, 
            gpu_id,
            batch_size=128):
    '''
    单卡运行

    Args:
        prototxt_path: deploy prototxt结构文件位置
        caffemodel_path: caffemodel模型参数文件位置
        lmdb_path: 测试lmdb文件位置
        cur_json_file: 当前GPU产生结果的输出文件
        gpu_num: 总的任务中运行GPU数量
        gpu_id: 运行的GPU id
        batch_size: 单批次运行样本数量，需要和deploy中数目保持一致
    '''
    print("GPU num: %d, cur gpu is %d gpu id" % (gpu_num, gpu_id))
    # set caffe gpu mode and gpu id
    # 设置GPU运行
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    # load caffe model
    # 读入caffe模型
    net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    mean = np.array([104, 117, 123]) # mean
    mean = mean[:, np.newaxis, np.newaxis]

    # create lmdb cursor
    # 创建lmdb数据库游标
    lmdb_env = lmdb.open(lmdb_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.MultiLabelDatum()

    cur_result = []

    cnt = -1
    num_batch = 0
    img_names = []
    img_labels = []
    img_id = []
    
    # 从数据库中读入数据
    for key, value in lmdb_cursor:
        # select samples correspending to gpu id
        # 读取样本的id，根据取摸运算分配GPU
        lmdb_id = lmdbkey_to_id(key)
        if lmdb_id % gpu_num != gpu_id:
            continue
        
        # print(key)
        cnt += 1
        img_id.append(lmdb_id)
        img_names.append(key[key.find('_')+1:])

        datum.ParseFromString(value)
        
        cur_gt_label = []
        for item in datum.multi_labels:
            cur_gt_label.append(item)
        img_labels.append(cur_gt_label)
        datamm = caffe.io.datum_to_array(datum.datum) 
        
        # print(datamm.shape, mean.shape)
        net.blobs['data'].data[cnt] = datamm[:,:224,:224] - mean
        
        # 如果达到批次要求数目
        if cnt == batch_size - 1:
            print('Batch ' + str(num_batch))
            num_batch += 1
            # forward for a batch of images
            # 执行模型前向计算，输出批次中样本概率
            out = net.forward() 
            
            # 将批次样本结果以及相关gt label，文件名等写入json结果文件
            for i in xrange(batch_size):
                output_prob = out['prob'][i]
                cur_sample = PredictResult(img_id[i], 
                        img_names[i], img_labels[i], output_prob.tolist())
                cur_result.append(cur_sample.to_dict())
            
            # set batch cnt to zero
            # 批次相关内容置0
            cnt = -1
            img_names = []
            img_labels = []
            img_id = []
            break

    # the last batch
    # 最后一个批次
    if cnt > 0:
        out = net.forward()
        for i in range(cnt):
            output_prob = out['prob'][i]
            cur_sample = PredictResult(img_id[i], 
                        img_names[i], img_labels[i], output_prob)                
            cur_result.append(cur_sample.to_dict())

    # write cur gpu cur result into file
    with open(cur_json_file, 'w') as f:
        json.dump(cur_result, f)

def main(prototxt_path, caffemodel_path, lmdb_path, cur_json_file, gpu_num, gpu_id):
    '''
    测试函数
    '''
    predict(prototxt_path, caffemodel_path, lmdb_path, cur_json_file, gpu_num, gpu_id)

if __name__ == "__main__":
    prototxt_path = sys.argv[1]
    caffemodel_path = sys.argv[2]
    lmdb_path = sys.argv[3]
    cur_json_file = sys.argv[4]
    gpu_num = int(sys.argv[5])
    gpu_id = int(sys.argv[6])
    print(prototxt_path, caffemodel_path, lmdb_path, cur_json_file, gpu_num, gpu_id)
    main(prototxt_path, caffemodel_path, lmdb_path, cur_json_file, gpu_num, gpu_id)
