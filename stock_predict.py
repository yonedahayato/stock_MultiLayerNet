# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import pickle
import sys

from get_make_data.get_stock_data import get_quote_yahoojp
from get_make_data.Granville_low import *
from MultiLayerNet.functions import *

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

    all_result = all_result.set_index('Date')
    all_result = all_result.sort_index()

    date = pd.DataFrame(all_result.index)
    all_result.index = range(len(all_result))
    all_result = pd.concat([date, all_result], axis=1)

    all_result.to_csv("get_predict/stock_data.csv")


def main():
    x = pd.read_csv("get_make_data/x_t_data/x.csv")
    t = pd.read_csv("get_make_data/x_t_data/t.csv")

    # hyper parameter の load
    with open("MultiLayerNet/params/best_HyperParams.pkl", "rb") as f:
        best_params_network = pickle.load(f)

    lr = best_params_network["lr"]
    weight_decay = best_params_network["weight_decay"]
    network = best_params_network["network"]

    # parameter の load
    network.load_params("best_Params.pkl")

    # code の情報を取得
    code = int(sys.argv[1]) # 6501
    #get_stock_data(code)

    all_stock_data = pd.read_csv("get_predict\stock_data.csv", encoding="shift-jis", index_col=0)

    # データの整形
    all_stock_data = all_stock_data.ix[:, ["Date", "code", "Close"]]
    all_stock_data = Make_NextDayData_TeachData(all_stock_data, "Close", NoTeach=True)
    all_stock_data = get_SMA(all_stock_data, "Close")
    x_diff = get_train_diff(all_stock_data, "Close")
    x_diff = x_diff.ix[len(x_diff)-1:,:]
    x_diff.index = [0]

    # get_MonthCodeの引数を作成する
    month_df = pd.DataFrame([str(datetime.date.today() - datetime.timedelta(days=1))], columns=["Date"])
    code_df = pd.DataFrame([code], columns=["code"])
    month_code = pd.concat([month_df, code_df], axis=1)

    month_code_dummy = get_MonthCode_DummyData(month_code)
    header = ["month_"+str(i+1) for i in range(12)]+["class_"+str(i+1) for i in range(9)]
    month_code_dummy = pd.concat([pd.DataFrame([], columns=header), month_code_dummy], axis=0)
    month_code_dummy = month_code_dummy.fillna(0) # Nan を 0に置換
    month_code_dummy = month_code_dummy.ix[:, header] # headerの順番に並べ替え

    x = pd.concat([month_code_dummy, x_diff], axis=1)
    x = np.array(x)

    # 次の日の株価（終値）が上がるか下がるかを予測
    predict_x = network.predict(x)
    y = softmax(predict_x)
    #print(y)
    if y[0,0]==0 and y[0,1]==1:
        print("上昇")
    elif y[0,0]==1 and y[0,1]==0:
        print("下降")
    else:
        print("error")


    return

if __name__ == '__main__':
    print("main")
    main()
