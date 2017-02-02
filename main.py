# coding: utf-8

import pandas as pd
import numpy as np
import datetime

from util import shuffle_dataset
from multi_layer_net import MultiLayerNet
from multi_layer_net_extend import MultiLayerNetExtend
from trainer import Trainer

def main():
    
    x = pd.read_csv("get_make_data/x_t_data/x.csv")
    t = pd.read_csv("get_make_data/x_t_data/t.csv")

    x = np.array(x)
    t = np.array(t)
    t = t.astype(np.uint8)

    x, t = shuffle_dataset(x, t)

    validation_rate = 0.20 # ハイパーパラメータ検証データは20%
    train_rate = 0.60 # 学習データは60%
    test_rate = 1 - (validation_rate + train_rate)

    validation_num = int(x.shape[0] * validation_rate)
    train_num = int(x.shape[0] * train_rate)

    x_val, t_val = x[:validation_num], t[:validation_num]
    x_train, t_train = x[validation_num:validation_num + train_num], t[validation_num:validation_num + train_num]
    x_test, t_test = x[validation_num + train_num:], t[validation_num + train_num:]

    def __train(lr, weight_decay, epocs=50): # epocs=50
        """
        network = MultiLayerNet(input_size=x.shape[1],
                                hidden_size_list=[100, 100, 100, 100, 100],
                                output_size=2,
                                weight_decay_lambda=weight_decay)
        """
        network = MultiLayerNetExtend(input_size=x.shape[1],
                                      hidden_size_list=[100, 100, 100, 100, 100],
                                      output_size=2,
                                      weight_decay_lambda=weight_decay,
                                      use_dropout=True,
                                      dropout_ration=0.5,
                                      use_batchnorm=True)

        batch_size = int(x_train.shape[0]/100)
        print("batch_size: "+str(batch_size))

        trainer = Trainer(network, x_train, t_train, x_val, t_val,
                          epochs=epocs,
                          mini_batch_size=batch_size, # 100
                          optimizer="sgd", # sgd
                          optimizer_param={"lr": lr},
                          verbose=False)

        trainer.train()

        Test_data_acc = network.accuracy(x_test, t_test)

        return trainer.test_acc_list, trainer.train_acc_list, Test_data_acc

    optimization_trial = 100

    results_val = {}
    results_train = {}
    results_test = {}

    for _ in range(optimization_trial):
        start = datetime.datetime.now()
        print(start)

        weight_decay = 10**np.random.uniform(-8, -2) # (-8, -4)
        lr = 10**np.random.uniform(-8, -1) # (-6, -2)

        val_acc_list, train_acc_list, test_acc = __train(lr, weight_decay)
        print("val acc:"+str(val_acc_list[-1])+", test acc:"+str(test_acc)+" | lr:"+str(lr)+", weight decay:"+str(weight_decay))

        key = "lr:"+str(lr)+", weight decay:"+str(weight_decay)

        results_val[key] = val_acc_list
        results_train[key] = train_acc_list
        results_test[key] = test_acc

        print("elapsed time:{0}".format(datetime.datetime.now() - start))

    print("=== Hyper-Parameter Optimization Result ===")
    i = 0
    for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
        print("Best-"+str(i+1)+"(val acc:"+str(val_acc_list[-1])+", test acc:"+str(results_test[key])+") | "+key)
        i+=1

        if i==20: break

    return

main()
