# -*- coding: utf-8 -*-

##################################################################
### 预测雾天的等级，共分为3级情况
##################################################################
import os
import sys
from svmutil import *
m = svm_load_model('Again_Again_Hour_Value_True_All.model')
print type(m) #输出模板M的类型
#读取SVM预测值
y, x = svm_read_problem('test.txt')
p_label, p_acc, p_val = svm_predict(y[:], x[:], m) #进行值预测
print "Predict value --->" + str(p_label)
##################################################################
### 预测是否存在雾天，则只需要改变m = svm_load_model('')即可
##################################################################



