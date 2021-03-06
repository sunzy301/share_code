from __future__ import print_function

import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

# prototxt file path
train_prototxt_path = './models/prototxts/sdnet_v200_train.prototxt'
test_prototxt_path = ''
deploy_prototxt_path = './models/prototxts/sdnet_v200_deploy.prototxt'

# solver prototxt
solver_prototxt_path = './models/prototxts/solver.prototxt'

# lmdb path
train_lmdb_path = ''
test_lmdb_path = ''

def netset(net, layer_name, layer):
    setattr(net, layer_name, layer);
    return getattr(net, layer_name);

def sdnet_v200():
    def conv_batch_relu(net, bottom, name, output, kernel, stride, pad, phase, with_relu=True):
        def conv_params(name):
            conv_kwargs = {
                'param': [{'name': name+'_w', 'lr_mult': 1, 'decay_mult': 1}, 
                            {'name': name+'_b', 'lr_mult': 2, 'decay_mult': 0}],
                'weight_filler': dict(type='msra'), 
                'bias_filler': dict(type='constant', value=0)
            }
            return conv_kwargs

        def bn_params(name, phase):
            bn_kwargs = {
                'use_global_stats': phase == caffe.TEST,
                'in_place': True,
                'param': [{"name": name+'_w', "lr_mult":0}, 
                            {"name": name+'_b', "lr_mult":0}, 
                            {"name": name+'_t', "lr_mult":0}]
            }
            return bn_kwargs

        def scale_params(name):
            scale_kwargs = {
                'in_place': True,
                'param': [{'name': name+'_w'}, {'name': name+'_b'}],
                'bias_term': True
            }
            return scale_kwargs

        conv_kwargs = conv_params(name+'_conv')
        bn_kwargs = bn_params(name+'_bn', phase)
        scale_kwargs = scale_params(name+'_scale')

        conv = netset(net, name+'_conv',
                    L.Convolution(bottom, kernel_size=kernel, stride=stride,
                                num_output=output, pad=pad, **conv_kwargs))
        batch = netset(net, name+'_bn',
                    L.BatchNorm(conv, **bn_kwargs))
        scale = netset(net, name+'_scale', 
                    L.Scale(batch, **scale_kwargs))
        if with_relu:
            relu = netset(net, name+'_relu', 
                    L.ReLU(scale, in_place=True))   
            return relu                
        else:
            return scale


    def inception_resnet_block(net, bottom, name, output_list, phase):
        # branch 1
        # 1 x 1 conv
        branch_1 = conv_batch_relu(net, bottom, name+'_1x1a', output_list[0][0], 1, 1, 0, phase)

        # branch 2
        # 1 x 1 conv
        branch_2 = conv_batch_relu(net, bottom, name+'_1x1b', output_list[1][0], 1, 1, 0, phase)
        # 3 x 3 conv
        branch_2 = conv_batch_relu(net, branch_2, name+'_3x3b', output_list[1][1], 3, 1, 1, phase)

        # branch 3
        # 1 x 1 conv
        branch_3 = conv_batch_relu(net, bottom, name+'_1x1c', output_list[2][0], 1, 1, 0, phase)
        # 3 x 3 conv
        branch_3 = conv_batch_relu(net, branch_3, name+'_3x3c', output_list[2][1], 3, 1, 1, phase)
        # 3 x 3 conv
        branch_3 = conv_batch_relu(net, branch_3, name+'_3x3c2', output_list[2][2], 3, 1, 1, phase)

        # branch 4
        # ave pooling
        branch_4 = netset(net, name+'_pool', L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=3, stride=1, pad=1))
        # 1 x 1 conv
        branch_4 = conv_batch_relu(net, branch_4, name+'_1x1d', output_list[3][0], 1, 1, 0, phase)

        # concat
        concat = netset(net, name+'_concat', L.Concat(branch_1, branch_2, branch_3, branch_4, axis=1))

        # res
        # 1 x 1 conv
        res = conv_batch_relu(net, concat, name+'_res_1x1', output_list[4][0], 1, 1, 0, phase, with_relu=False)

        # eltwise
        eltwise = netset(net, name+'_res_eltwise', L.Eltwise(res, bottom, operation=P.Eltwise.SUM))

        # relu
        relu = netset(net, name+'_res_relu', L.ReLU(eltwise, in_place=True))
        return relu

    def multipath_downsample_block(net, bottom, name, output_list, phase):
        # branch 1
        # 1 x 1 conv
        branch_1 = conv_batch_relu(net, bottom, name+'_1x1a', output_list[0][0], 1, 1, 0, phase)
        # 3 x 3 conv
        branch_1 = conv_batch_relu(net, branch_1, name+'_3x3a', output_list[0][1], 3, 2, 1, phase)

        # branch 2
        # 1 x 1 conv
        branch_2 = conv_batch_relu(net, bottom, name+'_1x1b', output_list[1][0], 1, 1, 0, phase)
        # 3 x 3 conv
        branch_2 = conv_batch_relu(net, branch_2, name+'_3x3b', output_list[1][1], 3, 1, 1, phase)
        # 3 x 3 conv
        branch_2 = conv_batch_relu(net, branch_2, name+'_3x3b2', output_list[1][2], 3, 2, 1, phase)

        # branch 3
        branch_3 = netset(net, name+'_pool', L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))

        # concat
        concat = netset(net, name+'_concat', L.Concat(branch_1, branch_2, branch_3,  axis=1))
        return concat

    def sdnet_v200_body(net, input_layer, num_cls, phase):
        # input image
        # 3 x 224 x 224

        # Group 1
        # conv1 
        # 64 x 112 x 112
        tensor = conv_batch_relu(net, input_layer, 'conv1', 64, 7, 2, 3, phase)
        # pool1
        # 64 x 56 x 56
        tensor = netset(net, 'pool1', L.Pooling(tensor, pool=P.Pooling.MAX, kernel_size=3, stride=2))

        # Group 2
        # conv2a
        # 64 x 56 x 56
        tensor = conv_batch_relu(net, tensor, 'conv2a', 64, 1, 1, 0, phase)
        # conv 2b
        # 128 x 56 x 56
        tensor = conv_batch_relu(net, tensor, 'conv2b', 128, 3, 1, 1, phase)
        # pool2
        tensor = L.Pooling(tensor, pool=P.Pooling.MAX, kernel_size=3, stride=2)

        # Group 3
        # inception resnet block 3a
        tensor = inception_resnet_block(net, tensor, 'in3a',
                                        [[32], [32, 32], [32, 48, 48], [32], [128]], phase)
        # inception resnet block 3b
        tensor = inception_resnet_block(net, tensor, 'in3b',
                                        [[32], [32, 48], [32, 48, 48], [32], [128]], phase)
        # multipath downsample block 3c                                        
        tensor = multipath_downsample_block(net, tensor, 'in3c',
                                        [[64, 80], [32, 48, 48]], phase)

        # Group 4
        # inception resnet block 4a
        tensor = inception_resnet_block(net, tensor, 'in4a',
                                        [[112], [32, 48], [48, 64, 64], [64], [256]], phase)
        # inception resnet block 4b
        tensor = inception_resnet_block(net, tensor, 'in4b',
                                        [[96], [48, 64], [48, 64, 64], [64], [256]], phase)
        # inception resnet block 4c
        tensor = inception_resnet_block(net, tensor, 'in4c',
                                        [[80], [64, 80], [64, 80, 80], [64], [256]], phase)
        # inception resnet block 4d
        tensor = inception_resnet_block(net, tensor, 'in4d',
                                        [[48], [64, 96], [80, 96, 96], [64], [256]], phase)
        # multipath downsample block 4e                                     
        tensor = multipath_downsample_block(net, tensor, 'in4e',
                                        [[64, 128], [96, 128, 128]], phase)

        # Group 5
        # inception resnet block 5a
        tensor = inception_resnet_block(net, tensor, 'in5a',
                                        [[176], [96, 160], [80, 112, 112], [64], [512]], phase)
        # inception resnet block 5b
        tensor = inception_resnet_block(net, tensor, 'in5b',
                                        [[176], [96, 160], [96, 112, 112], [64], [512]], phase)

        # Global Pooling
        tensor = netset(net, 'global_pooling', L.Pooling(tensor, pool=P.Pooling.AVE, kernel_size=7, stride=1, pad=0))

        # Full Connected  
        tensor = netset(net, 'main_fc', L.InnerProduct(tensor, num_output=num_cls, 
                        param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}],
                        weight_filler=dict(type='msra'),
                        bias_filler=dict(type='constant')))                                      
        return tensor
    
    # train net prototxt
    train_net = caffe.NetSpec()
    transform_param = dict(crop_size=224,
                           mean_value=[104, 117, 123],
                           mirror=True)
    data_param = dict(source=train_lmdb_path,
                        backend=P.Data.LMDB,
                        batch_size=64)                        
    train_net.data, train_net.label = L.MultilabelData(data_param=data_param,
                                    ntop=2,
                                    transform_param=transform_param,
                                    include=dict(phase=caffe.TRAIN)
    )
    logits = sdnet_v200_body(train_net, train_net.data, 20, caffe.TRAIN)
    train_net.loss = L.SigmoidWithLoss(logits, train_net.label)

    with open(train_prototxt_path, 'w') as f:
        print('name: "ROI"', file=f)
        print(train_net.to_proto(), file=f)

    # test net prototxt
    test_net = caffe.NetSpec()


    # deploy net prototxt
    deploy_net = caffe.NetSpec()
    deploy_net.data = L.Input(shape={"dim":[1, 3, 224, 224]})
    logits = sdnet_v200_body(deploy_net, deploy_net.data, 20, caffe.TEST)
    deploy_net.main_prob = L.Sigmoid(logits)

    with open(deploy_prototxt_path, 'w') as f:
        print('name: "ROI"', file=f)
        print(deploy_net.to_proto(), file=f)

def sdnet_v200_solver():
    s = caffe_pb2.SolverParameter()
    s.train_net = train_prototxt_path
    s.display = 100
    s.base_lr = 0.01
    s.lr_policy = "cosine"
    s.gamma = 0.1
    s.stepsize = 150000
    s.max_iter = 150000
    s.momentum = 0.9
    s.weight_decay = 0.0001
    s.snapshot = 1000
    s.snapshot_prefix = "/home/s00425426/ClassificBenchmark/result/train_log/all_20_sigmoid_0228/all_20_sigmoid_0228"
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    with open(solver_prototxt_path, 'w') as f:
        f.write(str(s))

def main():
    sdnet_v200()
    sdnet_v200_solver()

if __name__ == "__main__":
    main()
