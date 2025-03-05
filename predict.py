import keras
from model import Alexnet
import cv2
import numpy as np 
import utils

if __name__== "__main__":

    model = Alexnet.Alexnet()
    log_dir = "logs/last.h5"
    model.load_weights(log_dir)
    img = cv2.imread("test_img/test2.jpg")
    # print(img.shape)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = img1/255
    img1 = np.expand_dims(img1, axis=0)
    img1 = utils.resize_img(img1, (224,224))
    # print(img.shape)
    answer = utils.print_answer(np.argmax(model.predict(img1)))
    print(np.argmax(model.predict(img1)))
    # print("the answer is :",answer)
    # cv2.imshow("test_img", img)
    # cv2.waitKey(0)
