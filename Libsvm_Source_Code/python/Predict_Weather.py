#! /usr/bin/env python
# -*- coding: utf-8 -*-

#预测天气情况，且预测天气时，以最近的时间进行预测，即最新的时间对应的预测结果
import os
import sys
from svmutil import *
import MySQLdb #连接数据库
import numpy as np
import datetime


###根据事先已经求得的六个特征的最大值和最小值###



def GetLastTime_Data():#获取目前最新的相关数据，并进行预处理,连接数据库（通按时间排序，获取最新结果）
    
    #####################################################################################################
    #首先获取六大指标的相关参数,
    #对原始数据进行标准化处理，即具体形式如下：x`(i) = [x(i) - min( x(k) )] / [max( x(k) ) - min( x(k) )]
    #即在表weatherinformation中，执行此条类型语句
    Max_Temperature = 37.0
    Min_Temperature = 2.0
    Max_Min_Temperature = Max_Temperature - Min_Temperature
    Max_Humidity = 100.0
    Min_Humidity = 19.0
    Max_Min_Humidity = Max_Humidity - Min_Humidity
    Max_Visibility = 2000.0
    Min_Visibility = 27.0
    Max_Min_Visibility = Max_Visibility - Min_Visibility
    Max_WindSpeed = 15.1
    Min_WindSpeed = 0.0
    Max_Min_WindSpeed = Max_WindSpeed - Min_WindSpeed
    Max_WindDes = 359.9
    Min_WindDes = 0.0
    Max_Min_WindDes = Max_WindDes - Min_WindDes
    Max_Surface_Temperature = 59.4
    Min_Surface_Temperature = 2.9
    Max_Min_Surface_Temperature = Max_Surface_Temperature - Min_Surface_Temperature
    ######################################################################################################
    #对于已经处理后的表weatherinformation_Again中，并不需要处理预处理数据（标准化数据）
    #首先连接数据库
    conn = MySQLdb.connect(user="root", host="localhost", passwd="", db="WarningSystemDatabase")
    cur = conn.cursor()
    key = "max(GetTime)"
    tablename = "weatherinformation_Again"
    #首先进行最大时间值获取
    sql = "select " + key + " from " + tablename #获取最大数据的SQL语句
    #执行语句
    cur.execute(sql)
    #获取此刻时间<只有一条>
    row = cur.fetchone()
    lastGetTime = row[0] #获取此刻时间，且为str类型
    
    #之后获取相关数据，即六大特征
    key = "Temperature,Humidity,Visibility,WindSpeed,WindDes,Surface_Temperature"
    sql = "select " + key + " from " + tablename #获取相关值语句
    condition = " where GetTime=('%s')" % (lastGetTime.strftime('%Y-%m-%d %H:%M:%S'))#获取最新时间的数据条件
    sql = sql + condition
    #执行语句
    cur.execute(sql)
    #获取此刻时间的六大特征
    row = cur.fetchone()
    TemperatureString = row[0]
    HumidityString = row[1]
    VisibilityString = row[2]
    WindSpeedString = row[3]
    WindDesString = row[4]                                                
    Surface_TemperatureString = row[5]
    #之后对数据进行相关整理，以获得满足条件的结果！！！
    #构建天气参数字典
    weather_Parameter = []
    weather_Parameter.append(TemperatureString)
    weather_Parameter.append(HumidityString)
    weather_Parameter.append(VisibilityString)
    weather_Parameter.append(WindSpeedString)
    weather_Parameter.append(WindDesString)
    weather_Parameter.append(Surface_TemperatureString)
    #PS:weather_Parameter = [row[i] for i in range(6)] #简便方法构建
    key = [i + 1 for i in range(6)] #构建列表[1,2,3,4,5,6]
    #开始构建字典
    weather_Primary = dict(zip(key, weather_Parameter))
    x_predict = []
    x_predict.append(weather_Primary)
    y_predict = [-1] #默认情况下，预测未来时，值全设为-1，即无雾情况
    return y_predict, x_predict #返回相关的y,x值
    
#----------------------------------------------------------------------------------------------------------
# 重新定义此个函数，定义成预测文件中值,需要重新设置列表y,x
def GetLastTime_Data_File(x_list):
    key = [i + 1 for i in range(6)] 
    weather_Primary = dict(zip(key, x_list))
    x_predict = []
    x_predict.append(weather_Primary)
    y_predict = [-1]
    return y_predict, x_predict
#----------------------------------------------------------------------------------------------------------
    
def Predict_Fog(model):#输入预测模板，进行相关预测
    m = svm_load_model(model)
    #获取当前的预测值及其相关参数
    #y, x = svm_read_problem('test.txt')
    y, x = GetLastTime_Data() #进行赋值
    #对雾天情况进行值预测
    p_label, p_acc, p_val = svm_predict(y[:], x[:], m)
    p_label = int(p_label[0]) #每次只返回一个值
    #输出相关预测值
    print "Predict value --->" + str(p_label) #关于雾天情况的预测结果
    return p_label #并且返回相关预测结果（PS：对于是否有雾情况预测时，1表示有雾，-1表示无雾）
                   #对于雾天等级的判断时候，1表示一级情况（0～50能见度），
                   #2表示二级情况（50～200能见度），3表示三级情况（200～200能见度）
#----------------------------------------------------------------------------------------------------------
def Predict_Fog_File(model, x_list):#输入预测模板，进行相关预测
    m = svm_load_model(model)
    #获取当前的预测值及其相关参数
    #y, x = svm_read_problem('test.txt')
    y, x = GetLastTime_Data_File(x_list) #进行赋值
    #对雾天情况进行值预测
    p_label, p_acc, p_val = svm_predict(y[:], x[:], m)
    p_label = int(p_label[0]) #每次只返回一个值
    #输出相关预测值
    print "Predict value --->" + str(p_label) #关于雾天情况的预测结果
    return p_label #并且返回相关预测结果（PS：对于是否有雾情况预测时，1表示有雾，-1表示无雾）
                   #对于雾天等级的判断时候，1表示一级情况（0～50能见度），
                   #2表示二级情况（50～200能见度），3表示三级情况（200～200能见度）
#----------------------------------------------------------------------------------------------------------
                  
def Predict_Half_Hour_IS_Fog():#预测半个小时之后是否存在雾天情况
    model = 'F:\Python27\libsvm-3.20\libsvm-3.20\python\Again_Again_Half_Hour_all.model' #预测半个小时之后是否存在雾天情况的模板
    IS_Fog = Predict_Fog(model) #预测是否存在雾天情况
    #将值转化为Int类型
    #IS_Fog = int(IS_Fog)
    if IS_Fog == 1:#表示存在雾天，则进一步判断雾天等级
        Fog_Rank = Predict_Half_Hour_Fog_Rank()
        return Fog_Rank #返回雾天等级
    else:#即IS_Fog = -1情况时，即不存在雾天
        return IS_Fog #返回-1，即不存在雾天情况
# File----------------------------------------------------------------------------------------------------
def Predict_Half_Hour_IS_Fog_File(x_list):#预测半个小时之后是否存在雾天情况
    model = 'F:\Python27\libsvm-3.20\libsvm-3.20\python\Again_Again_Half_Hour_all.model' #预测半个小时之后是否存在雾天情况的模板
    IS_Fog = Predict_Fog_File(model, x_list) #预测是否存在雾天情况
    #将值转化为Int类型
    #IS_Fog = int(IS_Fog)
    if IS_Fog == 1:#表示存在雾天，则进一步判断雾天等级
        Fog_Rank = Predict_Half_Hour_Fog_Rank_File(x_list)
        return Fog_Rank #返回雾天等级
    else:#即IS_Fog = -1情况时，即不存在雾天
        return IS_Fog #返回-1，即不存在雾天情况
#----------------------------------------------------------------------------------------------------------   

def Predict_Half_Hour_Fog_Rank():#预测半个小时之后存在雾天等级
    model = 'F:\Python27\libsvm-3.20\libsvm-3.20\python\Again_Again_Half_Hour_Value_True_all.model'#预测半个小时之后存在雾天等级的模板
    Fog_Rank = Predict_Fog(model) #预测雾天等级情况
    return Fog_Rank #返回相关等级
# File----------------------------------------------------------------------------------------------------
def Predict_Half_Hour_Fog_Rank_File(x_list):#预测半个小时之后存在雾天等级
    model = 'F:\Python27\libsvm-3.20\libsvm-3.20\python\Again_Again_Half_Hour_Value_True_all.model'#预测半个小时之后存在雾天等级的模板
    Fog_Rank = Predict_Fog_File(model, x_list) #预测雾天等级情况
    return Fog_Rank #返回相关等级
#----------------------------------------------------------------------------------------------------------

def Predict_Hour_IS_Fog():#预测一个小时之后是否存在雾天情况
    model = 'F:\Python27\libsvm-3.20\libsvm-3.20\python\Again_Again_Hour_All.model' #预测一个小时之后是否存在雾天情况的模板
    IS_Fog = Predict_Fog(model) #预测是否存在雾天情况
    if IS_Fog == 1:#表示存在雾天，则进一步判断雾天等级
        Fog_Rank = Predict_Hour_Fog_Rank()
        return Fog_Rank #返回雾天等级
    else:#即IS_Fog = -1情况时，即不存在雾天
        return IS_Fog #返回-1，即不存在雾天情况

def Predict_Hour_Fog_Rank():#预测一个小时之后存在雾天等级
    model = 'F:\Python27\libsvm-3.20\libsvm-3.20\python\Again_Again_Hour_Value_True_All.model' #预测一个小时之后存在雾天等级的模板
    Fog_Rank = Predict_Fog(model) #预测雾天等级情况
    return Fog_Rank #返回相关等级


def test():
    
    #test读取数据情况
    y, x = GetLastTime_Data() #进行赋值
    print y
    print type(y)
    print x
    print type(x)
    print Predict_Half_Hour_IS_Fog()
    print Predict_Half_Hour_Fog_Rank()
