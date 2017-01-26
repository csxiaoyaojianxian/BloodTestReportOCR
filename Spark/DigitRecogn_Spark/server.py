# -*- coding: UTF-8 -*-
from __future__ import print_function
import BaseHTTPServer
import json
import csv
import shutil
from pyspark import SparkContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils

#服务器端配置
HOST_NAME = 'localhost'
PORT_NUMBER = 9000
reader = csv.reader(file('./data.csv', 'rb'))
output = open('LabeledPointsdata.txt', 'a')
reader = csv.reader(file('./data.csv', 'rb'))
output = open('LabeledPointsdata.txt', 'w')
n = 0

sc = SparkContext(appName="PythonNaiveBayesExample")

for line in reader:
    outputline ='%d' % int(n/500) + "," #每500行为一个数字的训练集
    n = n + 1
    for c in line:
        if "0.0000000000"==c:
            outputline += '0 '
        else:
            outputline += '1 '
    outputline += '\n'
    output.write(outputline)
output.close()
print('Format Successful!')

class JSONHandler(BaseHTTPServer.BaseHTTPRequestHandler):

    """处理接收到的POST请求"""
    def do_POST(self):
        response_code = 200
        response = ""
        var_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(var_len);
        payload = json.loads(content);

        # 如果是训练请求，训练然后保存训练完的神经网络
        if payload.get('train'):
            # 转化数据格式
            TrainData = ""
            for d in payload['trainArray'][0]['y0']:
                TrainData = TrainData + " " + ('%d' % d)
            TrainData = '%d' % (payload['trainArray'][0]['label']) + "," + TrainData.lstrip() +"\n"
            print(TrainData)
            Addoutput = open('LabeledPointsdata.txt', 'a')
            Addoutput.write(TrainData)
            Addoutput.close()


        # 如果是预测请求，返回预测值
        elif payload.get('predict'):
            try:
                training = MLUtils.loadLabeledPoints(sc, "LabeledPointsdata.txt")
                print('Begin NaiveBayes tranning!')
                model = NaiveBayes.train(training, 1.0)
                print('Trainning over!')
                print(payload['image'])
                response = {"type":"test", "result":str(model.predict(payload['image']))}
            except:
                response_code = 500
        else:
            response_code = 400

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if response:
            self.wfile.write(json.dumps(response))
        return

if __name__ == '__main__':
    server_class = BaseHTTPServer.HTTPServer;
    httpd = server_class((HOST_NAME, PORT_NUMBER), JSONHandler)

    try:
        #启动服务器
        httpd.serve_forever()
        print("Server started.")
    except KeyboardInterrupt:
        pass
    else:
        print ("Unexpected server exception occurred.")
    finally:
        httpd.server_close()

