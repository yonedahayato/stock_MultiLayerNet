# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import pickle
import sys

from get_make_data.get_stock_data import get_quote_yahoojp
from get_make_data.Granville_low import *

#from MultiLayerNet.util import shuffle_dataset
#from MultiLayerNet.multi_layer_net import MultiLayerNet
#from MultiLayerNet.multi_layer_net_extend import MultiLayerNetExtend
#from MultiLayerNet.trainer import Trainer

def get_stock_data(code):
    print(code)
    today = datetime.date.today() - datetime.timedelta(days=1)

    all_result = None
    cnt = 0
    for i in range(50):
        result = get_quote_yahoojp(code, str(today), str(today))

        if len(result)==0:
            today = today - datetime.timedelta(days=1)
            continue
        else:
            all_result = pd.concat([all_result, result], axis=0)
            today = today - datetime.timedelta(days=1)
            cnt += 1

        if cnt == 30: break

    all_result.index = range(len(all_result))
    all_result.to_csv("get_predict/stock_data.csv")


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

    #get_stock_data(code)

    all_stock_data = pd.read_csv("get_predict/stock_data.csv", encoding="shift-jis", index_col=0)
    print(all_stock_data)
    return

    all_stock_data = all_stock_data.ix[:, ["Date", "code", "Close"]]

    all_stock_data = Make_NextDayData_TeachData(all_stock_data, "Close")
    all_stock_data = get_SMA(all_stock_data, "Close")

    train_diff = get_train_diff(all_stock_data, "Close")
    teach_diff = get_teach_diff(all_stock_data)

    print(all_stock_data)
    month_code_dummy = get_MonthCode_DummyData(all_stock_data)

    train = pd.concat([month_code_dummy, train_diff], axis=1)
    teach = teach_diff
    
    return


    # 次の日の株価（終値）が上がるか下がるかを予測
    pass


main()
