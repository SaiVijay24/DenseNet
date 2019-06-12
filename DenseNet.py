#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:34:34 2019

@author: cy
"""
import tensorflow as tf 
import numpy as np 
import model3 as M 
from tflearn.layers.conv import global_avg_pool

class denseNet(M.Model):
    def initialize(self,embedding_size, embedding_bn=True,outchn=0):
        self.first_cn=M.ConvLayer(7, 32, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.dense_1=Dense_block(4,32)
        self.dense_2=Dense_block(4,64)
        self.dense_3=Dense_block(4,128)
        self.dense_4=Dense_block(4,256)
        self.bn = M.BatchNorm()
        self.transition=M.ConvLayer(1,256, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
         #self.transition=M.ConvLayer(6, 2*outchn, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.embedding = M.Dense(embedding_size, batch_norm=embedding_bn)
    def forward(self,x):
        print(x.shape,"is the shape of the input")
        x=self.first_cn(x)
        x=M.ConvLayer(1,int(x.shape[-1]/2), activation=M.PARAM_LRELU, usebias=False, batch_norm=True)(x)
        x=self.dense_1(x)
        print(x.shape,"is the shape after the dense_1")
        x=M.ConvLayer(1,int(x.shape[-1]/2), activation=M.PARAM_LRELU, usebias=False, batch_norm=True)(x)
        print(x.shape,"is the shape of the transition shape")
        x=self.dense_2(x)
        x=M.ConvLayer(1,int(x.shape[-1]/2), activation=M.PARAM_LRELU, usebias=False, batch_norm=True)(x)
        x=self.dense_3(x)
        x=M.ConvLayer(1,int(x.shape[-1]/2), activation=M.PARAM_LRELU, usebias=False, batch_norm=True)(x)
        x=self.dense_4(x)
        x=self.first_cn(x)
        x=self.bn(x)
        x = M.flatten(x)
        x = tf.nn.dropout(x,0.4)
        x = self.embedding(x)
        return x


class Dense_block(M.Model): # 
    def initialize(self,nb_layers,filters):
        self.nb_layers=nb_layers
        self.filters=filters
        self.bottle_neck=Bottle_Neck_layer(self.filters)
    def Concatenation(self,layers):
        return tf.concat(layers, axis=3)
    def forward(self,input_x):
        layers_concat = list()
        layers_concat.append(input_x)
        x=self.bottle_neck(input_x)
        #print(x.shape,"shape of the bottle neck")
        layers_concat.append(x)
        for i in range(self.nb_layers-1):
            x=self.Concatenation(layers_concat)
            x=self.bottle_neck(x)
            layers_concat.append(x)
        #print(x.shape,"after the looping block")
        x = self.Concatenation(layers_concat)
        return x


class Bottle_Neck_layer(M.Model):
    def initialize(self, outchn):
        self.c1 = M.ConvLayer(1, 4*outchn, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
        self.c2=M.ConvLayer(3, outchn, activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
    def forward(self,x):
        x=self.c1(x)
        x=self.c2(x)
        return x
    
     # BN -> PRELU -> conv 1*1 [4x channels]-> BN -> conv 3*3 [x channel]