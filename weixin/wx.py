# coding: utf-8
import sys
sys.path.append("..")
import web
import hashlib
import urllib2
import time
from lxml import etree
from PIL import Image
import BloodTestReportOCR.tf_predict
from BloodTestReportOCR.imageFilter import ImageFilter
import numpy, cv2
import json
urls = (
    '/weixin', 'Weixin'
)

token = "galigeigei"

class Weixin:
    def __init__(self):
        self.render = web.template.render('./')

    def POST(self):
        str_xml = web.data()
        xml = etree.fromstring(str_xml)
        msgType = xml.find("MsgType").text
        fromUser = xml.find("FromUserName").text
        toUser = xml.find("ToUserName").text

        res = '请输入图片'
        if msgType == 'image':
            print('gali')
            url = xml.find('PicUrl').text
            img = cv2.imdecode(numpy.fromstring(urllib2.urlopen(url).read(), numpy.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
            data = ImageFilter(image=img).ocr(22)
            if data:
                data = json.loads(data)
                pre = [str(data['bloodtest'][i]['value']) for i in range(22)]
                for i in range(22):
                    if pre[i] == '': pre[i] = 0
                    else:
                        tmp = pre[i].replace('.', '', pre[i].count('.')-1)
                        pre[i] = float(tmp)
                       
                arr = numpy.array(pre)
                arr = numpy.reshape(arr, [1, 22])
                
                sex, age = tf_predict.predict(arr)
                res = 'sex:'+['女','男'][sex] + '  age:'+str(int(age))
            else:
                res = '请输入正确图片'

        return self.render.reply_text(fromUser, toUser, int(time.time()), res)

    def GET(self):
        data = web.input()
        signature = data.signature
        timestamp = data.timestamp
        nonce = data.nonce
        echostr = data.echostr
        list = [token, timestamp, nonce]
        list.sort()
        str = list[0] + list[1] + list[2]
        hashcode = hashlib.sha1(str).hexdigest()
        if hashcode == signature: return echostr

app = web.application(urls, globals())

if __name__ == '__main__':  
    app.run()
