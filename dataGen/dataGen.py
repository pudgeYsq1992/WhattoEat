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
    return

def genRanddata(worksheet):
    for i in range(2,101):
        if random.randint(1,4) = 1:
            worksheet.write(i,1,'')
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