import os
import random
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

ImageSize = 64


def resize_image(image, height=ImageSize, width=ImageSize):
    top, bottom, left, right = (0, 0, 0, 0)

    h, w, _ = image.shape
    # print(h, w, _)

    longest = max(h, w)
    # print(longest)

    if h < longest:
        dh = longest - h
        top = dh // 2
        bottom = dh - top
    elif w < longest:
        dw = longest - w
        left = dw // 2
        right = dw - left
    else:
        pass

    Black = [0, 0, 0]

    # 图片填充，top,botto,left,right指向上下左右补全长度，value指填充颜色
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # cv2.imshow("img",constant)
    # cv2.waitKey(0)

    return cv2.resize(constant, (height, width))


images = []
labels = []


def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))  # 返回绝对路径
        if os.path.isdir(full_path):  # isdir()判断是否为文件夹，是文件夹的话继续递归调用
            read_path(full_path)  # 递归调用
        else:
            if dir_item.endswith('.jpg'):
                img = cv2.imread(full_path)
                img = resize_image(img, ImageSize, ImageSize)

                images.append(img)  # 文件夹下的所有图片
                labels.append(path_name)  # 标签为文件夹名字

    return images, labels


def load_dataset(path_name):
    images, labels = read_path(path_name)

    images = np.array(images)
    # print(images.shape)

    labels = np.array([0 if label.endswith('Hinton') else 1 for label in labels])

    return images, labels


# print(read_path("D:/desktop/face/"))
images, labels = load_dataset("D:/desktop/face/")



