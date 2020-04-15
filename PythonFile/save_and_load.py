#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import json
import datetime


def get_name(location):
    return str(location)+str('\\')+str(datetime.datetime.now().date())+'.json'

def get_name_setting(location):
    return str(location)+str('\\setting_')+str(datetime.datetime.now().date())+'.json'


def history2json(history_):
    history_['Normalization'][0] = history_['Normalization'][0].tolist()
    history_['Normalization'][1] = history_['Normalization'][1].tolist()
    for i in range(len(history_['params'])):
        history_['params'][i]['W'] = history_['params'][i]['W'].tolist()
        history_['params'][i]['b'] = history_['params'][i]['b'].tolist()
    return history_


def json2history(json_):
    json_['Normalization'][0] = np.array(json_['Normalization'][0],dtype=np.float32)
    json_['Normalization'][1] = np.array(json_['Normalization'][1],dtype=np.float32)
    for i in range(len(json_['params'])):
        json_['params'][i]['W'] = np.array(json_['params'][i]['W'],dtype=np.float64)
        json_['params'][i]['b'] = np.array(json_['params'][i]['b'],dtype=np.float64)
    return json_


def save_(json_,location):    # 将json数据保存在location内（日期.json）
    with open(get_name(location), 'w') as json_file:
        json.dump(json_, json_file, ensure_ascii=False)


def save_setting(json_,location):    # 将json数据保存在location内（日期.json）
    with open(get_name_setting(location), 'w') as json_file:
        json.dump(json_, json_file, ensure_ascii=False)


def load_(location):    # 将模型文件（日期.json）从location提取出来
    with open(location, 'r') as json_file:
        json_ = json.load(json_file)
        # json.load(json_, json_file, ensure_ascii=False)
    return  json_

if __name__ == '__main__':
    # print(get_name())

    # test_js={'a':5,'b':[1,2,3,4]}
    # save_(test_js,r'C:\jupyter file\json')

    diction = load_(r'C:\jupyter file\json\data.json')
    print(type(diction))
    print(diction['Normalization'])
    a=json2history(diction)
    print(a)
