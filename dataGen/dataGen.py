# -*- coding: utf-8 -*-
import xlrd
import xlwt
import os
import operator
import string
from string import Template
from xlutils.copy import copy
import re
import random

def writeColName(worksheet):
    worksheet.write(1,1,'天气')
    worksheet.write(1,2,'疲劳程度')
    worksheet.write(1,3,'周几')
    worksheet.write(1,4,'店家口味')
    worksheet.write(1,5,'菜价')
    worksheet.write(1,6,'距公司距离')
    worksheet.write(1,8,'你有多想去这家店？')
    return

def genRanddata(worksheet):
    #生成“天气”一列
    for i in range(2,101):
        randomData = random.randint(0,3)
        if randomData == 0:
            worksheet.write(i,1,'恶劣')
        if randomData == 1:
            worksheet.write(i,1,'较差')
        if randomData == 2:
            worksheet.write(i,1,'较好')
        if randomData == 3:
            worksheet.write(i,1,'很好')
    #生成“疲劳程度”一列   
    for i in range(2,101):
        randomData = random.randint(0,3)
        if randomData == 0:
            worksheet.write(i,2,'很累')
        if randomData == 1:
            worksheet.write(i,2,'有点累')
        if randomData == 2:
            worksheet.write(i,2,'不大累')
        if randomData == 3:
            worksheet.write(i,2,'很轻松')
    #生成“周几”一列   
    for i in range(2,101):
        randomData = random.randint(0,6)
        if randomData == 0:
            worksheet.write(i,3,'星期一')
        if randomData == 1:
            worksheet.write(i,3,'星期二')
        if randomData == 2:
            worksheet.write(i,3,'星期三')
        if randomData == 3:
            worksheet.write(i,3,'星期四')
        if randomData == 4:
            worksheet.write(i,3,'星期五')
        if randomData == 5:
            worksheet.write(i,3,'星期六')
        if randomData == 6:
            worksheet.write(i,3,'星期天')
    #生成“店家口味”一列   
    for i in range(2,101):
        randomData = random.randint(0,3)
        if randomData == 0:
            worksheet.write(i,4,'不好吃')
        if randomData == 1:
            worksheet.write(i,4,'一般般')
        if randomData == 2:
            worksheet.write(i,4,'还不错')
        if randomData == 3:
            worksheet.write(i,4,'很好吃')
    #生成“菜价”一列   
    for i in range(2,101):
        randomData = random.randint(0,3)
        if randomData == 0:
            worksheet.write(i,5,'便宜')
        if randomData == 1:
            worksheet.write(i,5,'还行')
        if randomData == 2:
            worksheet.write(i,5,'有点贵')
        if randomData == 3:
            worksheet.write(i,5,'很贵')
    #生成“距离”一列   
    for i in range(2,101):
        randomData = random.randint(0,3)
        if randomData == 0:
            worksheet.write(i,6,'很远')
        if randomData == 1:
            worksheet.write(i,6,'有些远')
        if randomData == 2:
            worksheet.write(i,6,'不算远')
        if randomData == 3:
            worksheet.write(i,6,'很近')
    return
    
def dataGen():
    #data = open('待标注数据.xlsx' ,'w')	
    #data.close()
    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet('1')
    writeColName(ws)
    genRanddata(ws)
    #wb.sheets()[0].cell(0,0).value =  'helloworld' 
    wb.save(r'待标注数据.xls')

    return

if __name__ == '__main__':	
    #任务开始
    dataGen()

    print("data generation complete!")