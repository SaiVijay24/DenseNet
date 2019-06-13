#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:34:19 2019

@author: cy
"""


import tensorflow as tf 
import numpy as np 
import model3 as M 

#def inception_v3(pretrained=False, progress=True, **kwargs):
#    r"""Inception v3 model architecture from
#    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
#    .. note::
#        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
#        N x 3 x 299 x 299, so ensure your images are sized accordingly.
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#        progress (bool): If True, displays a progress bar of the download to stderr
#        aux_logits (bool): If True, add an auxiliary branch that can improve training.
#            Default: *True*
#        transform_input (bool): If True, preprocesses the input according to the method with which it
#            was trained on ImageNet. Default: *False*
#    """
#    if pretrained:
#        if 'transform_input' not in kwargs:
#            kwargs['transform_input'] = True
#        if 'aux_logits' in kwargs:
#            original_aux_logits = kwargs['aux_logits']
#            kwargs['aux_logits'] = True
#        else:
#            original_aux_logits = True
#        model = Inception3(**kwargs)
#        state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
#                                              progress=progress)
#        model.load_state_dict(state_dict)
#        if not original_aux_logits:
#            model.aux_logits = False
#            del model.AuxLogits
#        return model
#
#    return Inception3(**kwargs)


class Inception3(M.Model):

    def initialize(self,embedding_size,embedding_bn=True,outchn=0):
        #super(Inception3, self).__init__()
        #self.aux_logits = aux_logits
        #self.transform_input = transform_input M.ConvLayer(7, 32, stride=2, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.Conv2d_1a_3x3 = M.ConvLayer(3, 32,activation=M.PARAM_LRELU, usebias=False, batch_norm=True, stride=2)
        self.Conv2d_2a_3x3 = M.ConvLayer(3, 32, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.Conv2d_2b_3x3 = M.ConvLayer(3, 64, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.Conv2d_3b_1x1 = M.ConvLayer(1, 80, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.Conv2d_4a_3x3 = M.ConvLayer(3, 192, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.Mixed_5b = InceptionA(pool_features=32) #192
        self.Mixed_5c = InceptionA(pool_features=64)#256
        self.Mixed_5d = InceptionA(pool_features=64)#288
        self.Mixed_6a = InceptionB()#288
        self.Mixed_6b = InceptionC(channels_7x7=128)#768
        self.Mixed_6c = InceptionC(channels_7x7=160)
        self.Mixed_6d = InceptionC(channels_7x7=160)
        self.Mixed_6e = InceptionC(channels_7x7=192)
        #if aux_logits:
            #self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE()
        self.Mixed_7c = InceptionE()
        self.fc = M.Dense(embedding_size, batch_norm=embedding_bn)

#        for m in self.modules():
#            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                import scipy.stats as stats
#                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
#                X = stats.truncnorm(-2, 2, scale=stddev)
#                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
#                values = values.view(m.weight.size())
#                with torch.no_grad():
#                    m.weight.copy_(values)
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.constant_(m.weight, 1)
#                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = M.MaxPool(3,2)(x) #(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x =  M.MaxPool(3,2)(x) #F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
#        if self.training and self.aux_logits:
#            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        #x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = M.flatten(x)
        #x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = tf.nn.dropout(x,0.4)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
#        if self.training and self.aux_logits:
#            return _InceptionOuputs(x, aux)
        return x


class InceptionA(M.Model):
   #M.ConvLayer(3, 32,activation=M.PARAM_LRELU, usebias=False, batch_norm=True, stride=2)
    def initialize(self,pool_features):
        
        self.branch1x1 = M.ConvLayer(1, 64,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)

        self.branch5x5_1 = M.ConvLayer(1, 48,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch5x5_2 = M.ConvLayer(5,64,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)

        self.branch3x3dbl_1 = M.ConvLayer(1,64,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch3x3dbl_2 = M.ConvLayer(1,96,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch3x3dbl_3 =  M.ConvLayer(3,96,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)

        self.branch_pool = M.ConvLayer(1,pool_features,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)  #__init__(self, size, stride, pad='SAME')  __init__(self, size, stride, pad='SAME')
        self.avg_pool=M.AvgPool(3,1)
    def Concatenation(self,layers):
        return tf.concat(layers, axis=3)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return self.Concatenation(outputs)


class InceptionB(M.Model):

    def initialize(self):
        
        self.branch3x3 = M.ConvLayer(3, 384,activation=M.PARAM_LRELU, usebias=False, batch_norm=True,stride=2) #M.ConvLayer(3, 384,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)

        self.branch3x3dbl_1 = M.ConvLayer(1, 64,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch3x3dbl_2 = M.ConvLayer(3, 96,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch3x3dbl_3 = M.ConvLayer(3, 96,activation=M.PARAM_LRELU, usebias=False, batch_norm=True,stride=2)  #__init__(self, size, stride, pad='SAME')
        self.pool=M.MaxPool(3,2)
    def Concatenation(self,layers):
        return tf.concat(layers, axis=3)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.pool(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return self.Concatenation(outputs)


class InceptionC(M.Model):

    def initialize(self,channels_7x7):
        
        self.branch1x1 = M.ConvLayer(1, 192,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)#M.ConvLayer(1, 192,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)

        c7 = channels_7x7
        self.branch7x7_1 = M.ConvLayer(1, c7,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch7x7_2 = M.ConvLayer([1,7], c7,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch7x7_3 = M.ConvLayer([7,1],192,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)

        self.branch7x7dbl_1 = M.ConvLayer(1,c7,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch7x7dbl_2 = M.ConvLayer([7,1],c7,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch7x7dbl_3 = M.ConvLayer([1,7],c7,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch7x7dbl_4 = M.ConvLayer([7,1],c7,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch7x7dbl_5 = M.ConvLayer([1,7],192,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)

        self.branch_pool = M.ConvLayer(1,192,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.pool=M.MaxPool(3,1) #__init__(self, size, stride, pad='SAME')
    def Concatenation(self,layers):
        return tf.concat(layers, axis=3)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool =self.pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return self.Concatenation(outputs)



class InceptionD(M.Model):

    def initialize(self,channels_7x7):  #M.ConvLayer(1, 192,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch3x3_1 = M.ConvLayer(1, 192,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch3x3_2 = M.ConvLayer(3, 320, stride=2,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)

        self.branch7x7x3_1 = M.ConvLayer(1, 192,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch7x7x3_2 = M.ConvLayer([1,7], 192,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch7x7x3_3 = M.ConvLayer([7,1], 192,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch7x7x3_4 = M.ConvLayer(3, 192,stride=2,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.max_pool=M.MaxPool(3,2)  #__init__(self, size, stride, pad='SAME')
    def Concatenation(self,layers):
        return tf.concat(layers, axis=3)


    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.max_pool(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return self.Concatenation(outputs)


class InceptionE(M.Model):

    def initialize(self):
        self.branch1x1 = M.ConvLayer(1,320,activation=M.PARAM_LRELU, usebias=False, batch_norm=True) #M.ConvLayer(1,320,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)

        self.branch3x3_1 = M.ConvLayer(1,384,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch3x3_2a = M.ConvLayer([1,3],384,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch3x3_2b = M.ConvLayer([3,1],384,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)

        self.branch3x3dbl_1 = M.ConvLayer(1,448,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch3x3dbl_2 =  M.ConvLayer(3,384,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch3x3dbl_3a =  M.ConvLayer([1,3],384,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.branch3x3dbl_3b =   M.ConvLayer([3,1],384,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)

        self.branch_pool = M.ConvLayer(1,192,activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.avg_pool=M.AvgPool(3,1)
    def Concatenation(self,layers):
        return tf.concat(layers, axis=3)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = self.Concatenation(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = self.Concatenation(branch3x3dbl)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return self.Concatenation(outputs)


#class InceptionAux(nn.Module):
#
#    def __init__(self, in_channels, num_classes):
#        super(InceptionAux, self).__init__()
#        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
#        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
#        self.conv1.stddev = 0.01
#        self.fc = nn.Linear(768, num_classes)
#        self.fc.stddev = 0.001
#
#    def forward(self, x):
#        # N x 768 x 17 x 17
#        x = F.avg_pool2d(x, kernel_size=5, stride=3)
#        # N x 768 x 5 x 5
#        x = self.conv0(x)
#        # N x 128 x 5 x 5
#        x = self.conv1(x)
#        # N x 768 x 1 x 1
#        # Adaptive average pooling
#        x = F.adaptive_avg_pool2d(x, (1, 1))
#        # N x 768 x 1 x 1
#        x = x.view(x.size(0), -1)
#        # N x 768
#        x = self.fc(x)
#        # N x 1000
#        return x


#class BasicConv2d(nn.Module):
#
#    def __init__(self, in_channels, out_channels, **kwargs):
#        super(BasicConv2d, self).__init__()
#        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
#        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
#
#    def forward(self, x):
#        x = self.conv(x)
#        x = self.bn(x)
#        return F.relu(x, inplace=True)


