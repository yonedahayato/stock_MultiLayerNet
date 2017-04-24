# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import pickle
import sys
import random

from MultiLayerNet.util import shuffle_dataset
from MultiLayerNet.multi_layer_net import MultiLayerNet
from MultiLayerNet.multi_layer_net_extend import MultiLayerNetExtend
from MultiLayerNet.trainer import Trainer

from train_method import *

def __train(lr, weight_decay,
            x, x_val, x_train, x_test,
            t, t_val, t_train, t_test,
            epocs=50,# epocs=50
            Optimizer="sgd",
            hidden_size_list=[100, 100, 100, 100, 100]):
    """
    network = MultiLayerNet(input_size=x.shape[1],
                            hidden_size_list=[100, 100, 100, 100, 100],
                            output_size=2,
                            weight_decay_lambda=weight_decay)
    """
    network = MultiLayerNetExtend(input_size=x.shape[1],
                                  hidden_size_list=hidden_size_list,
                                  output_size=1, # 分類（判別）の場合は2
                                  weight_decay_lambda=weight_decay,
                                  use_dropout=True,
                                  dropout_ration=0.5,
                                  use_batchnorm=True)

    # レイアーの確認
    #network.layer_cheack()

    batch_size = int(x_train.shape[0]/100)
    print("batch_size: "+str(batch_size))
    print("Optimizer: {}".format(Optimizer))

    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs,
                      mini_batch_size=batch_size, # 100
                      optimizer=Optimizer, # sgd
                      optimizer_param={"lr": lr},
                      verbose=False) # verbose...loss, acc, etc を表示する

    trainer.train()

    # パラメーターの保存
    #network.save_params("params.pkl")
    #network.load_params("params.pkl")

    Test_data_acc = network.accuracy(x_test, t_test, regression=True) # 正解率
    # Test_data_loss = network.loss(x_test, t_test, real_mean=True) # 変化率の分散

    # ...return...
    # ...trainer.test_acc_list...val_acc_list
    # ...trainer.train_acc_list...train_acc_list
    # ...Test_data_acc...test_acc
    # ...network...network

    return trainer.test_acc_list, trainer.train_acc_list, Test_data_acc, network

def main():
    x = pd.read_csv("get_make_data/x_t_data/x.csv")
    t = pd.read_csv("get_make_data/x_t_data/t.csv")

    x = np.array(x)
    t = np.array(t)
    #t = t.astype(np.uint8) #分類（判別）のときは必要

    x, t = shuffle_dataset(x, t)
    m = 4 # データの1/mを使用
    x_len, t_len = int(len(x)/m), int(len(t)/m)
    x, t = x[:x_len], t[:t_len]

    validation_rate = 0.20 # ハイパーパラメータ検証データは20%
    train_rate = 0.60 # 学習データは60%
    test_rate = 1 - (validation_rate + train_rate)

    validation_num = int(x.shape[0] * validation_rate)
    train_num = int(x.shape[0] * train_rate)

    x_val, t_val = x[:validation_num], t[:validation_num]
    x_train, t_train = x[validation_num:validation_num + train_num], t[validation_num:validation_num + train_num]
    x_test, t_test = x[validation_num + train_num:], t[validation_num + train_num:]

    #---#
    # ハイパーパラメータを検証する回数
    optimization_trial = 10 # 100 
    """
    Optimizer = "Adam" # 最適化法 {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                       #           'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}

    best_params_network = hyper_parameter_verification(optimization_trial, Optimizer,
                                                       x, x_val, x_train, x_test,
                                                       t, t_val, t_train, t_test) # hyper parameterの検証
    # ノードの数と最適化法を固定してハイパーパラメータ(leaning rate, weight decay)の検証を行う
    """
    """
    Optimizer_dict = {"sgd":"SGD", "adam":"Adam"}
    optimizer_verification(Optimizer_dict, optimization_trial,
                           x, x_val, x_train, x_test,
                           t, t_val, t_train, t_test)
    # ハイパーパラメーター(learning rate, weight decay)とノードの数 を固定して optimizer(最適化法)毎に評価する
    """
    """
    unit_verification(x, x_val, x_train, x_test,
                      t, t_val, t_train, t_test)
    """
    """
    grid_search(x, x_val, x_train, x_test,
                t, t_val, t_train, t_test)
    """
    print("bayesian_optimizer \n")
    bayesian_optimizer()
    sys.exit()

    # best parameter の save
    best_params_network["network"].save_params("best_Params_{}.pkl".fromat(Optimizer))

    # hyper parameter の save
    with open("MultiLayerNet/params/best_HyperParams_{}.pkl".fromat(Optimizer), "wb") as f:
        pickle.dump(best_params_network, f)

    return


if __name__ == '__main__':
    print("main_train_main")
    print("\n")
    main()
