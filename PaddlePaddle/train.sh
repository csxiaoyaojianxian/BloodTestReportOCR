
set -e
config=vgg.py
output=./vgg_model
log=train.log

paddle train \
--config=$config \
--use_gpu=0 \
--trainer_count=8 \
--num_passes=10 \
--save_dir=$output \
2>&1 | tee $log

python -m paddle.utils.plotcurve -i $log > plot.png

:<<'
use_gpu是否使用GPU训练
trainer_count训练线程数，使用CPU时建议设为CPU的线程数，使用GPU时设为GPU的数目
num_passes训练次数,每训练一次会生成一个模型文件夹
output模型存放路径
'

