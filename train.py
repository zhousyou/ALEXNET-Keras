import keras
import numpy as np 
from model import Alexnet
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical
import cv2

def generate_files(lines, batch_size):
    """处理数据，打包成batchsize,输出生成器"""
    num = len(lines)
    i = 0
    while True:
        trainX = []
        trainY = []

        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(";")[0]
            img = cv2.imread("data/image/train/"+name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255
            img = cv2.resize(img,(224,224))
            trainX.append(img)
            trainY.append(lines[i].split(";")[1])
        trainX = np.array(trainX).reshape(-1,224,224,3)
        trainY = to_categorical(trainY, num_classes=2)
        yield trainX, trainY



if __name__=="__main__":

    log_dir =  "logs/"
    
    with open('data/dataset.txt', "r") as f:
        lines = f.readlines()

    #读取数据总数
    nums = len(lines)  #25000
    # print(nums)
    np.random.seed(1)
    np.random.shuffle(lines)

    #90%用于训练，10%用于验证
    train_nums = int(nums*0.9)
    test_nums = nums - train_nums
    batch_size = 128

    model = Alexnet.Alexnet()

    #保存的方式
    checkpoint = ModelCheckpoint(filepath= log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val-loss{val_loss.3f}.h5',
                                 monitor='acc',
                                 save_best_only=True,
                                 save_weights_only=True,
                                 period=3)
    
    earlystopping = EarlyStopping(monitor='acc',
                                  min_delta=0,
                                  patience=10,
                                  verbose=1)
    reduce_learningrate = ReduceLROnPlateau(monitor='acc',
                                            factor=0.5,
                                            patience=3,
                                            verbose=1)
                                            
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(generator=generate_files(lines[:train_nums], batch_size), 
                        steps_per_epoch=train_nums//batch_size,
                        epochs=50,validation_data=generate_files(lines[train_nums:], batch_size),
                        validation_steps=test_nums//batch_size,
                        callbacks=[checkpoint, earlystopping, reduce_learningrate])
    model.save_weights("logs/" + 'last.h5')
