
set -e
data_dir=./data
python preprocess.py -i $data_dir -s 28 -c 0

#-i后为训练数据存放路径，-s后为图像大小,-c后为图像有没有颜色
