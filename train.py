import keras
import numpy as np 


if __name__=="__main__":

    log_dir =  "logs/"
    
    with open('data/dataset.txt', "r") as f:
        lines = f.readlines()

    #读取数据总数
    nums = len(lines)  #25000
    # print(nums)


    #90%用于训练，10%用于验证
    train_nums = int(nums*0.9)
    test_nums = nums - train_nums

