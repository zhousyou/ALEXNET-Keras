import keras
from model import Alexnet
import cv2

if __name__== "__main__":

    model = Alexnet.Alexnet()
    log_dir = "logs/last.h5"
    model.load_weights(log_dir)
    img = cv2.imread("test_img/test2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255
    img.resize(224,224)

    print(model.predict(img))
