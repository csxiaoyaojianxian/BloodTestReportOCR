# -*- coding: UTF-8 -*-
import cv2

def  digitsimg(src):
    
    #灰度化
    img_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

    #Otsu thresholding 二值化
    ret,result= cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #腐蚀去除一些小的点
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,2))
    eroded = cv2.erode(result,kernel)

    #将结果放大便于识别
    result = cv2.resize(result,(128,128),interpolation=cv2.INTER_CUBIC)

   # cv2.imshow('result',result)
   # cv2.waitKey(0)

    #腐蚀去除放大后的一些小的点
    eroded = cv2.erode(result,kernel)
  #  cv2.imshow('eroded',eroded)
  #  cv2.waitKey(0)
    #膨胀使数字更饱满
    result = cv2.dilate(eroded,kernel)
 #   cv2.imshow('dilated',result)

    #直方图均衡化使图像更清晰
    cv2.equalizeHist(result)
    #中值滤波去除噪点
    result = cv2.medianBlur(result,5)
#    cv2.imshow('median',result)
 #   cv2.waitKey(0)
    return result
'''
def chineseimg(src):

    

    #灰度化
    img_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)


    #Otsu thresholding 二值化
    ret,result= cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  #  cv2.imshow('otsu',result)
  #  cv2.waitKey(0)


    #直方图均衡化使图像更清晰
    cv2.equalizeHist(result)
  #  cv2.imshow('直方图',result)
 #   cv2.waitKey(0)
    return result

    #将结果放大便于识别
    result = cv2.resize(result,(256,128),interpolation=cv2.INTER_CUBIC)

    #腐蚀去除放大后的一些小的点
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,2))
    eroded = cv2.erode(result,kernel)
    cv2.imshow('eroded',eroded)
    cv2.waitKey(0)

    #膨胀使数字更饱满
    result = cv2.dilate(eroded,kernel)
    cv2.imshow('dilated',result)
    cv2.waitKey(0)

    #直方图均衡化使图像更清晰
    cv2.equalizeHist(result)
    #中值滤波去除噪点
    result = cv2.medianBlur(result,5)
    cv2.imshow('median',result)
    cv2.waitKey(0)'''
    

