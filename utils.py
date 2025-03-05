import numpy as np 
import cv2

def resize_img(img, size):
    images = []

    for i in img:
        i = cv2.resize(i, size)
        images.append(i)
    images = np.array(images)
    return images

def print_answer(argmax):

    with open("model/index_word.txt","r", encoding='utf-8') as f:
        index = [i.split(';')[1][:-1] for i in f.readlines()]
    return index[argmax]