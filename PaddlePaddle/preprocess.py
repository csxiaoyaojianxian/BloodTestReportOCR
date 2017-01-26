# -*- coding: UTF-8 -*-
from paddle.utils.preprocess_img import ImageClassificationDatasetCreater
from optparse import OptionParser

#处理命令行参数
def option_parser():
    parser = OptionParser(usage="usage: python preprcoess.py "\
                          "-i data_dir [options]")
    parser.add_option("-i", "--input", action="store",
                      dest="input", help="图片路径")
    parser.add_option("-s", "--size", action="store",
                      dest="size", help="图片大小")
    parser.add_option("-c", "--color", action="store",
                      dest="color", help="图片有没有颜色")
    return parser.parse_args()

if __name__ == '__main__':
     options, args = option_parser()
     data_dir = options.input
     processed_image_size = int(options.size)
     color = options.color == "1"
     data_creator = ImageClassificationDatasetCreater(data_dir,
                                                      processed_image_size,
                                                      color)
     #每个训练文件包含的图片数
     data_creator.num_per_batch = 1000
     data_creator.overwrite = True
     data_creator.create_batches()
