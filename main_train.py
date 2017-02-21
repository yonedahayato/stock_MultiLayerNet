# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import pickle

from MultiLayerNet.util import shuffle_dataset
from MultiLayerNet.multi_layer_net import MultiLayerNet
from MultiLayerNet.multi_layer_net_extend import MultiLayerNetExtend
from MultiLayerNet.trainer import Trainer

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

    Test_data_acc = network.accuracy(x_test, t_test, regression=True)

    return trainer.test_acc_list, trainer.train_acc_list, Test_data_acc, network

def hyper_parameter_verification(optimization_trial, Optimizer,
                                 x, x_val, x_train, x_test,
                                 t, t_val, t_train, t_test):
    results_val = {}
    results_train = {}
    results_test = {}

    # hyper_parameter
    params_dict_lr = {} # leaning_rate
    params_dict_WeightDecay = {} # weight_decay

    network_dict = {}


    for _ in range(optimization_trial):
        start = datetime.datetime.now()
        print(start)

        weight_decay = 10**np.random.uniform(-8, -4) # (-8, -4)
        lr = 10**np.random.uniform(-6, -2) # (-6, -2)

        params_dict = {}

        val_acc_list, train_acc_list, test_acc, network = __train(lr, weight_decay,
                                                                  x, x_val, x_train, x_test,
                                                                  t, t_val, t_train, t_test,
                                                                  Optimizer=Optimizer) #学習

        print("val acc:"+str(val_acc_list[-1])+", test acc:"+str(test_acc)+" | lr:"+str(lr)+", weight decay:"+str(weight_decay))

        key = "lr:"+str(lr)+", weight decay:"+str(weight_decay)

        results_val[key] = val_acc_list
        results_train[key] = train_acc_list
        results_test[key] = test_acc

        params_dict_lr[key] = lr
        params_dict_WeightDecay[key] = weight_decay

        network_dict[key] = network

        print("elapsed time:{0}".format(datetime.datetime.now() - start))

    print("=== Hyper-Parameter Optimization Result ===")
    i = 0
    best_params_network = {}
    for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
        print("Best-"+str(i+1)+"(val acc:"+str(val_acc_list[-1])+", test acc:"+str(results_test[key])+") | "+key)
        print(params_dict_lr[key])
        print(params_dict_WeightDecay[key])
        if i==0:
            best_params_network["lr"] = params_dict_lr[key]
            best_params_network["weight_decay"] = params_dict_WeightDecay[key]
            best_params_network["network"] = network_dict[key]

        i+=1
        if i==10: break

    return best_params_network

def optimizer_verification(Optimizer_dict, optimization_trial,
                           x, x_val, x_train, x_test,
                           t, t_val, t_train, t_test):
    results_val = {}
    results_train = {}
    results_test = {}

    network_dict = {}
    Optimizer_point = {"sgd":0, "adam":0}

    for _ in range(optimization_trial):
        for key_opt, value in Optimizer_dict.items():
            start = datetime.datetime.now()
            print(start)
            weight_decay = 10**np.random.uniform(-8, -4)
            lr = 10**np.random.uniform(-6, -2)

            Optimizer_dict_result = {}

            val_acc_list, train_acc_list, test_acc, network = __train(lr, weight_decay,
                                                                      x, x_val, x_train, x_test,
                                                                      t, t_val, t_train, t_test,
                                                                      Optimizer=key_opt) #学習

            print("val acc:"+str(val_acc_list[-1])+", test acc:"+str(test_acc)+" | Optimizer:"+str(key_opt))

            key = str(key_opt)

            results_val[key] = val_acc_list
            results_train[key] = train_acc_list
            results_test[key] = test_acc

            Optimizer_dict[key] = str(key_opt)

            network_dict[key] = network

            print("elapsed time:{0}".format(datetime.datetime.now() - start))

        if results_test["sgd"] < results_test["adam"]:
            Optimizer_point["adam"] += 1
        elif results_test["sgd"] > results_test["adam"]:
            Optimizer_point["sgd"] += 1
        else:
            pass
    print("====Optimizer_point====")
    print("sgd: {}, adam: {}".format(Optimizer_point["sgd"], Optimizer_point["adam"]))



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




    optimization_trial = 5 # 100
    Optimizer = "Adam" # 最適化法 {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                       #           'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}

    # hyper_parameter
    params_dict_lr = {} # leaning_rate
    params_dict_WeightDecay = {} # weight_decay

    network_dict = {}
    """
    best_params_network = hyper_parameter_verification(optimization_trial, Optimizer,
                                                       x, x_val, x_train, x_test,
                                                       t, t_val, t_train, t_test) # hyper parameterの検証
    """
    
    Optimizer_dict = {"sgd":"SGD", "adam":"Adam"}
    optimizer_verification(Optimizer_dict, optimization_trial,
                           x, x_val, x_train, x_test,
                           t, t_val, t_train, t_test)
    return



    # best parameter の save
    best_params_network["network"].save_params("best_Params_{}.pkl".fromat(Optimizer))

    # hyper parameter の save
    with open("MultiLayerNet/params/best_HyperParams_{}.pkl".fromat(Optimizer), "wb") as f:
        pickle.dump(best_params_network, f)

    return


if __name__ == '__main__':
    print("main_train_main")
    main()
