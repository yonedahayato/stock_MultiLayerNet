# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import pickle
import sys

from MultiLayerNet.util import shuffle_dataset
from MultiLayerNet.multi_layer_net import MultiLayerNet
from MultiLayerNet.multi_layer_net_extend import MultiLayerNetExtend
from MultiLayerNet.trainer import Trainer

def main():
    x = pd.read_csv("get_make_data/x_t_data/x.csv")
    t = pd.read_csv("get_make_data/x_t_data/t.csv")

    # hyper parameter の load
    with open("MultiLayerNet/best_HyperParams.pkl", "rb") as f:
        best_params_network = pickle.load(f)

    lr = best_params_network["lr"]
    weight_decay = best_params_network["weight_decay"]
    network = best_params_network["network"]

    # parameter の load
    network.load_params("best_Params.pkl")

    # code の情報を取得
    code = int(sys.argv[1])
    print(code)
    


    # 次の日の株価（終値）が上がるか下がるかを予測
    pass


main()
