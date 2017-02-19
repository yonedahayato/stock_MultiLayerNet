# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import pickle
import sys

from get_make_data.get_stock_data import get_quote_yahoojp, get_Kdb
from get_make_data.Granville_low import *
from MultiLayerNet.functions import *

#from MultiLayerNet.util import shuffle_dataset
#from MultiLayerNet.multi_layer_net import MultiLayerNet
#from MultiLayerNet.multi_layer_net_extend import MultiLayerNetExtend
#from MultiLayerNet.trainer import Trainer

def get_stock_data_ByYahoojp(code):
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
    print(all_result)

    #all_result.to_csv("get_predict/stock_data.csv")

def get_stock_data_ByKdb(code):
    base = "http://k-db.com/stocks/{}-T?download=csv".format(int(code))
    all_result = get_Kdb(code,"-",base)

    all_result = all_result.ix[len(all_result)-30:,:]
    all_result.index = range(len(all_result))
    print(all_result)

    #all_result.to_csv("get_predict/stock_data.csv")

def main():
    x = pd.read_csv("get_make_data/x_t_data/x.csv")
    t = pd.read_csv("get_make_data/x_t_data/t.csv")

    # hyper parameter の load
    print("hyper parameter loading...")
    with open("MultiLayerNet/params/best_HyperParams.pkl", "rb") as f:
        best_params_network = pickle.load(f)

    lr = best_params_network["lr"]
    weight_decay = best_params_network["weight_decay"]
    network = best_params_network["network"]

    # parameter の load
    print("parameter loading...")
    network.load_params("best_Params.pkl")

    # スコアの確認
    n = 20
    score = network.predict(np.array(x))[:n,:]
    teach = np.array(t)[:n,:]
    
    print(["score","teach"])
    print(np.c_[score,teach])
    score[score>0], score[score==0], score[score<0] = 1, 0, -1
    teach[teach>0], teach[teach==0], teach[teach<0] = 1, 0, -1
    print(np.c_[score,teach])
    accuracy = np.sum(score == teach) / float(score.shape[0])
    print(accuracy)
    #return ###---###

    # best parameter の正解率を確認
    best_accuracy = network.accuracy(np.array(x), np.array(t), regression=True)
    print("best_accuracy: {}".format(best_accuracy))
    return

    # code の情報を取得
    code = int(sys.argv[1]) # 6501
    #get_stock_data_ByYahoojp(code) # 最近の情報を取得
    get_stock_data_ByKdb(code)
    return ###---###

    all_stock_data = pd.read_csv("get_predict\stock_data.csv", encoding="shift-jis", index_col=0)

    new_data = all_stock_data.ix[len(all_stock_data)-1:,:] # 最新のデータ
    new_data.index = range(len(new_data))
    #print(new_data)

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

    # 次の日の株価（終値）がどの程度上がるか下がるかを予測
    predict_x = network.predict(x)
    # y = softmax(predict_x) # 分類（判別）の場合は使用する

    predict_value = int(new_data.ix[0,"Close"]*(1+predict_x[0,0]))
    Type="Close"

    # if y[0,0]==0 and y[0,1]==1:
    if predict_x[0,0]>0:
        print("上昇 {}:{}".format(Type, predict_value))
    # elif y[0,0]==1 and y[0,1]==0:
    elif predict_x[0,0]<0:
        print("下降 {}:{}".format(Type, predict_value))
    else:
        #print("error")
        print("変化なし")
    return

if __name__ == '__main__':
    print("stock_predict_main")
    main()
