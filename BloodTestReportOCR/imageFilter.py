# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import os
import json
import classifier
import math
try:
    from PIL import Image
except ImportError:
    from PIL import Image
import pytesseract
# import Binarization
import imgproc
from PIL import Image

default = [3, 70, 30, 0.7, 0.0001]

class ImageFilter:
    
    '''
        ImageFilter初始化，可以传入一个opencv格式打开的图片，也可以传入一个图片的路径，二选一
    '''
    def __init__(self, image, imagepath='origin_pics/bloodtestreport2.jpg'):
        self.img = image
        if image is None:
            print 'img init from',imagepath
            self.img = cv2.imread(imagepath)

        self.PerspectiveImg = None
         #设置输出路径，创建目录
        self.output_path = 'temp_pics/'
        if not(os.path.exists(self.output_path)):
            os.makedirs(self.output_path)

    '''
        perspect函数用于透视image，他会缓存一个透视后的opencv numpy矩阵，并返回该矩阵
        透视失败，则会返回None，并打印不是报告
        @param 透视参数
    '''
    def perspect(self, param=default):
        #载入参数
        gb_param = param[0] #必须是奇数
        canny_param_upper = param[1]
        canny_param_lower = param[2]
        ref_lenth_multiplier = param[3]
        ref_close_multiplier = param[4]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))

        # 载入图像，灰度化，开闭运算，描绘边缘
        
        img_sp = self.img.shape
        ref_lenth = img_sp[0] * img_sp[1] * ref_lenth_multiplier
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_gb = cv2.GaussianBlur(img_gray, (gb_param, gb_param), 0)
        closed = cv2.morphologyEx(img_gb, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(opened, canny_param_lower , canny_param_upper)

        # 调用findContours提取轮廓
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        def getbox(i):
            rect = cv2.minAreaRect(contours[i])
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            return box

        def distance(box):
            delta1 = box[0]-box[2]
            delta2 = box[1]-box[3]
            distance1 = np.dot(delta1,delta1)
            distance2 = np.dot(delta2,delta2)
            distance_avg = (distance1 + distance2) / 2
            return distance_avg

        # 筛选出对角线足够大的几个轮廓
        found = []
        for i in range(len(contours)):
            box = getbox(i)
            distance_arr = distance(box)
            if distance_arr > ref_lenth:
                found.append([i, box])

        def getline(box):
            if np.dot(box[1]-box[2],box[1]-box[2]) < np.dot(box[0]-box[1],box[0]-box[1]):
                point1 = (box[1] + box[2]) / 2
                point2 = (box[3] + box[0]) / 2
                lenth = np.dot(point1-point2, point1-point2)
                return point1, point2, lenth
            else:
                point1 = (box[0] + box[1]) / 2
                point2 = (box[2] + box[3]) / 2
                lenth = np.dot(point1-point2, point1-point2)
                return point1, point2, lenth

        def cmp(p1, p2):
            delta = p1 - p2
            distance = np.dot(delta, delta)
            if distance < img_sp[0] * img_sp[1] * ref_close_multiplier:
                return 1
            else:
                return 0

        def linecmp(l1, l2):
            f_point1 = l1[0]
            f_point2 = l1[1]
            f_lenth = l1[2]
            b_point1 = l2[0]
            b_point2 = l2[1]
            b_lenth = l2[2]
            if cmp(f_point1,b_point1) or cmp(f_point1,b_point2) or cmp(f_point2,b_point1) or cmp(f_point2,b_point2):
                if f_lenth > b_lenth:
                    return 1
                else:
                    return -1
            else:
                return 0

        def deleteline(line, j):
            lenth = len(line)
            for i in range(lenth):
                if line[i] is j:
                    del line[i]
                    return

        # 将轮廓变为线
        line = []

        for i in found:
            box = i[1]
            point1, point2, lenth = getline(box)
            line.append([point1, point2, lenth])

        # 把不合适的线删去
        if len(line)>3:
            for i in line:
                for j in line:
                    if i is not j:
                        rst = linecmp(i, j)
                        if rst > 0:
                            deleteline(line, j)
                        elif rst < 0:
                            deleteline(line, i)

        #检测出的线数量不对就返回-1跳出
        if len(line) != 3:
            print "it is not a is Report!,len(line) =",len(line)
            return None
        
        def distance_line(i, j):
            dis1 = np.dot(i[0]-j[0], i[0]-j[0])
            dis2 = np.dot(i[0]-j[1], i[0]-j[1])
            dis3 = np.dot(i[1]-j[0], i[1]-j[0])
            dis4 = np.dot(i[1]-j[1], i[1]-j[1])
            return min(dis1, dis2, dis3, dis4)

        def findhead(i, j, k):
            dis = []
            dis.append([distance_line(i, j), i, j])
            dis.append([distance_line(j, k), j, k])
            dis.append([distance_line(k, i), k, i])
            dis.sort()
            if dis[0][1] is dis[2][2]:
                return dis[0][1], dis[2][1]
            if dis[0][2] is dis[2][1]:
                return dis[0][2], dis[2][2]

        def cross(vector1, vector2):
            return vector1[0]*vector2[1]-vector1[1]*vector2[0]

        # 由三条线来确定表头的位置和表尾的位置
        line_upper, line_lower = findhead(line[2],line[1],line[0])

	def detectmiss(line, line_lower, ref_angle):
	    vector = []
	    j = 0
	    if linecmp(line[1], line_lower):
	        j = 1
	    elif linecmp(line[2], line_lower):
	        j = 2

	    lenth = len(line)
	    for i in range(lenth):
	        if i != j:
		    vector.append([line[j][0]-line[i][0], line[j][1]-line[i][1]])
	    vect1 = vector[0][0]
	    vect2 = vector[0][1]
	    vect3 = vector[1][0]
	    vect4 = vector[1][1]
	    angle1 = (math.acos(np.dot(vect3, vect1) / ((np.dot(vect1, vect1) ** 0.5) * (np.dot(vect3, vect3)**0.5))))/math.pi*180
	    angle2 = (math.acos(np.dot(vect4, vect2) / ((np.dot(vect2, vect2) ** 0.5) * (np.dot(vect4, vect4)**0.5))))/math.pi*180
	    if angle1 > ref_angle or angle2 > ref_angle:
	        return 1
	    return 0

	# 通过计算夹角来检测是否有缺失一角的情况
	ref_angle = 1
	if detectmiss(line, line_lower, ref_angle):
	    print "it is not a complete Report!"
	    return None

        # 由表头和表尾确定目标区域的位置

        # 利用叉乘的不可交换性确定起始点
        total_width = line_upper[1]-line_upper[0]
        total_hight = line_lower[0]-line_upper[0]
        cross_prod = cross(total_width, total_hight)
        if cross_prod <0:
            temp = line_upper[1]
            line_upper[1] = line_upper[0]
            line_upper[0] = temp
            temp = line_lower[1]
            line_lower[1] = line_lower[0]
            line_lower[0] = temp

        #由于需要表格外的数据，所以变换区域需要再向上和向下延伸
        left_axis = line_lower[0] - line_upper[0]
        right_axis = line_lower[1] - line_upper[1]
        line_upper[0] = line_upper[0] - left_axis * 2 / 15
        line_upper[1] = line_upper[1] - right_axis * 2 / 15
        line_lower[0] = line_lower[0] + left_axis * 2 / 15
        line_lower[1] = line_lower[1] + right_axis * 2 / 15

        #设定透视变换的矩阵
        points = np.array([[line_upper[0][0], line_upper[0][1]], [line_upper[1][0], line_upper[1][1]], 
                        [line_lower[0][0], line_lower[0][1]], [line_lower[1][0], line_lower[1][1]]],np.float32)
        standard = np.array([[0,0], [1000, 0], [0, 760], [1000, 760]],np.float32)

        #使用透视变换将表格区域转换为一个1000*760的图
        PerspectiveMatrix = cv2.getPerspectiveTransform(points,standard)
        self.PerspectiveImg = cv2.warpPerspective(self.img, PerspectiveMatrix, (1000, 760))

        #输出透视变换后的图片
        cv2.imwrite(self.output_path + 'region.jpg', self.PerspectiveImg)
        return self.PerspectiveImg
        
    '''
        filter函数返回img经过透视过后的PIL格式的Image对象，如果缓存中有PerspectivImg则直接使用，没有先进行透视
        过滤失败则返回None
        @param filter参数
    '''
    def filter(self, param=default):
        if self.PerspectiveImg is None:
            self.PerspectivImg = self.perspect(param)
            if self.PerspectiveImg is None:
                return None
        img = Image.open(self.output_path + 'region.jpg')
        if not(classifier.isReport(img)):
            print "it is not a is Report!",classifier.isReport(self.PerspectiveImg)
            return None
        try:
            Image.fromarray(self.PerspectivImg)
        except Exception:
            return None
        return Image.fromarray(self.PerspectivImg)

    '''
        autocut函数用于剪切ImageFilter中的img成员，剪切之后临时图片保存在out_path，
        如果剪切失败，返回-1，成功返回0
        @num 剪切项目数
        @param 剪切参数
    '''
    def autocut(self, num, param=default):
        if self.PerspectiveImg is None:
            self.PerspectivImg = self.filter(param)
        # 仍然是空，说明不是报告
        if self.PerspectiveImg is None:
            return -1

        #输出年龄
        img_age = self.PerspectiveImg[15 : 70, 585 : 690]
        cv2.imwrite(self.output_path + 'age.jpg', img_age)

        #输出性别
        img_gender = self.PerspectiveImg[15 : 58, 365 : 420]
        cv2.imwrite(self.output_path + 'gender.jpg', img_gender)

        #输出时间
        img_time = self.PerspectiveImg[722 : 760, 430 : 630]
        cv2.imwrite(self.output_path + 'time.jpg', img_time)

        #转换后的图分辨率是已知的，所以直接从这个点开始读数据就可以了
        startpoint = [199, 132]
        vertical_lenth = 37
        lateral_lenth = 80

        def getobjname(i, x, y):
            region_roi = self.PerspectiveImg[y : y+vertical_lenth, x : x+170]
            filename = self.output_path + 'p' + str(i) + '.jpg'
            cv2.imwrite(filename, region_roi)

        def getobjdata(i, x, y):
            region_roi = self.PerspectiveImg[y : y+vertical_lenth, x : x+lateral_lenth]
            filename = self.output_path + 'data' + str(i) + '.jpg'
            cv2.imwrite(filename, region_roi)

        #输出图片
        if num <= 13 and num > 0:
            for i in range(num):
                getobjname(int(i), 25, startpoint[1])
                getobjdata(int(i), startpoint[0], startpoint[1])
                startpoint[1] = startpoint[1] + 40
        elif num > 13:
            for i in range(13):
                getobjname(int(i), 25, startpoint[1])
                getobjdata(int(i), startpoint[0], startpoint[1])
                startpoint[1] = startpoint[1] + 40
            startpoint = [700, 135]
            for i in range(num-13):
                getobjname(int(i+13), 535, startpoint[1])
                getobjdata(int(i+13), startpoint[0], startpoint[1])
                startpoint[1] = startpoint[1] + 40
                

        #正常结束返回0
        return 0
    
    '''
        ocr函数用于对img进行ocr识别，他会先进行剪切，之后进一步做ocr识别，返回一个json对象
        如果剪切失败，则返回None
        @num 规定剪切项目数

    '''
    def ocr(self, num):
        digtitsresult = []
        chiresult = []
        # 不是报告
        if self.autocut(num) == -1:
            return None
        # 识别
        def image_to_string(image, flag=True):
            if flag:
                text = pytesseract.image_to_string(Image.fromarray(image), config='-psm 7 digits')
            else:
                text = pytesseract.image_to_string(Image.fromarray(image), lang='chi_sim', config=' -psm 7 Bloodtest')
            return text

        # 读取图片
        def read(url):
            image = cv2.imread(url)
            return image
        # load json example
        with open('bloodtestdata.json') as json_file:
            data = json.load(json_file)

        # 识别检测项目编号及数字
        for i in range(num):
            item = read('temp_pics/p' + str(i) + '.jpg')
            item_num = classifier.getItemNum(item)
            image = read('temp_pics/data' + str(i) + '.jpg')
            image = imgproc.digitsimg(image)
            digtitstr = image_to_string(image)
            digtitstr = digtitstr.replace(" ", '')
            digtitstr = digtitstr.replace("-", '')
            digtitstr = digtitstr.strip(".")
            data['bloodtest'][item_num]['value'] = digtitstr
        json_data = json.dumps(data,ensure_ascii=False,indent=4)
        return json_data


# unit test
if __name__ == '__main__':
    imageFilter = ImageFilter() # 可以传入一个opencv格式打开的图片
    
    num = 22
    print imageFilter.ocr(num)

