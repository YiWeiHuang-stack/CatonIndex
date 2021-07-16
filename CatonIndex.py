import threading
from tqdm import tqdm
import cv2
import warnings
import os
import numpy as np
warnings.filterwarnings("ignore")

path = 'D:/experiment/AVFDU/origin' # 图片路径
img_load_size = [256,256] # 图象初始化尺寸

# 亮度均衡化
def histeq(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    histeq = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    histeq[..., 0] = cv2.equalizeHist(histeq[..., 0])
    return cv2.cvtColor(histeq, cv2.COLOR_YUV2BGR)

def diff(i0,i1):
    return cv2.absdiff(i0,i1).mean()

LabData = [os.path.join(path,f) for f in os.listdir(path)] #记录文件名用
frames = [histeq(cv2.resize(cv2.imread(f),(img_load_size[0],img_load_size[1]))) for f in LabData]

sum_variance = 0
mean = 0
for pos in range(0, len(LabData) - 2, 3):
    f0 = frames[pos]
    f1 = frames[pos+1]
    f2 = frames[pos+2]
    d0 = diff(f0,f1)
    d1 = diff(f1,f2)
    avg = (d0+d1) / 2
    x = ((d0 - avg)**2 + (d1 - avg)**2) / 2
    sum_variance += x
    mean += 1
mean = sum_variance / mean
print(sum_variance,mean)
