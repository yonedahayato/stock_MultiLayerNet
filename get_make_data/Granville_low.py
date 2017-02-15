# -*- coding: utf_8 -*-  

import numpy as np
import pandas as pd
from collections import OrderedDict
from datetime import datetime as dt

def data_load(variable, file_name):
    data = pd.read_csv(file_name, encoding="shift-jis", index_col=0)
    data.index = range(len(data))

    return data.ix[:, variable]

def Make_NextDayData_TeachData(all_stock_data, variable, NoTeach=False):
    brand_code_list = list(pd.read_csv("get_make_data/tosyo1.csv").ix[:,"code"])

    all_stock_data_tmp = None
    for brand_code in brand_code_list: # 銘柄ごと
        stock_data = all_stock_data.query("code == @brand_code")

        if len(stock_data) == 0: continue

        stock_data.index = range(len(stock_data))

        N = 5 # n日分先のデータを変数とする
        new_variable_dic = {}
        for n in range(1,N+1):
            new_variable_dic["next-"+str(n)+variable] = stock_data.ix[n:, [variable]]
            new_variable_dic["next-"+str(n)+variable].index = range(len(new_variable_dic["next-"+str(n)+variable]))
            new_variable_dic["next-"+str(n)+variable].columns = ["next-"+str(n)+variable]

            stock_data = pd.concat([stock_data, new_variable_dic["next-"+str(n)+variable]], axis=1)

        if NoTeach == False:
            T = 2 # t日先の教師データ
            teach_data_dic = {}
            for t in range(1,T+1):
                teach_data_dic["teach-"+str(t)] = stock_data.ix[N+t:, ["Close"]] #教師データはcloseを使用
                teach_data_dic["teach-"+str(t)].index = range(len(teach_data_dic["teach-"+str(t)]))
                teach_data_dic["teach-"+str(t)].columns = ["teach-"+str(t)]

                stock_data = pd.concat([stock_data, teach_data_dic["teach-"+str(t)]], axis=1)

        stock_data = stock_data.dropna() # NaN削除
        all_stock_data_tmp = pd.concat([all_stock_data_tmp, stock_data], axis=0)

        del stock_data

    all_stock_data_tmp.index = range(len(all_stock_data_tmp))
    all_stock_data = all_stock_data_tmp

    del all_stock_data_tmp

    return all_stock_data

def Simple_Moving_Average(data):
    data.index = range(len(data))
    data_tmp = data

    data_name_list = list(data.columns)
    data_name = data_name_list[0]

    terms = [15, 10, 5]
    for term in terms:
        data = pd.concat([data, pd.rolling_mean(data_tmp, term)], axis=1)
        data_name_list.append("SMA-" + str(term)+"_"+data_name)

    data.columns = data_name_list
    return data

def get_SMA(all_stock_data, variable):

    brand_code_list = list(pd.read_csv("get_make_data/tosyo1.csv").ix[:,"code"])
    all_stock_data_tmp = None
    for brand_code in brand_code_list:
        stock_data = all_stock_data.query("code == @brand_code")
        stock_data.index = range(len(stock_data))

        close_data = stock_data.ix[:, [variable]]
        close_SMA_data = Simple_Moving_Average(close_data)

        stock_data = pd.concat([stock_data, close_SMA_data.ix[:,1:]], axis=1)
        stock_data = stock_data.dropna() # NaN削除

        all_stock_data_tmp = pd.concat([all_stock_data_tmp, stock_data], axis=0)

    all_stock_data_tmp.index = range(len(all_stock_data_tmp))

    return all_stock_data_tmp

def get_train_diff(all_stock_data, variable):
    variable_list = list(all_stock_data.columns)

    close_variable_list = []
    SMA_variable_list = []
    teach_variable_list = []
    else_variable_list = []
    for v in variable_list:
        if "SMA" in v:
            SMA_variable_list.append(v)

        elif ("Close" in v) and ("SMA" not in v):
            close_variable_list.append(v)

        elif "teach" in v:
            teach_variable_list.append(v)

        else:
            else_variable_list.append(v)

    train_diff = None
    for SMA_v in SMA_variable_list:
        for close_v in close_variable_list:
            #train_dict[SMA_v+"-"+close_v] = list(np.array(all_stock_data.ix[:,[SMA_v]]) - np.array(all_stock_data.ix[:,[close_v]]))
            diff_df = pd.DataFrame(all_stock_data.apply(lambda x: x[SMA_v] - x [close_v], axis=1))
            diff_df.columns = [SMA_v+"-"+close_v]

            train_diff = pd.concat([train_diff, diff_df], axis=1)

    return train_diff

def get_teach_diff(all_stock_data, numeric=False):
    if numeric:
        teach_diff = ( np.array(all_stock_data.ix[:, "teach-2"]) - np.array(all_stock_data.ix[:, "teach-1"]) ) / np.array(all_stock_data.ix[:, "teach-1"])
        return pd.DataFrame(teach_diff,columns=["teach"])

    teach_diff_mask = (np.array(all_stock_data.ix[:, "teach-2"]) - np.array(all_stock_data.ix[:, "teach-1"]))>0
    teach_diff = teach_diff_mask.astype(np.int64)

    return pd.DataFrame(teach_diff,columns=["teach"])

def get_MonthCode_DummyData(all_stock_data):
    month_data = all_stock_data.ix[:,["Date"]]
    code_data = all_stock_data.ix[:,["code"]]

    month_list = []
    code_list = []
    for i in range(len(all_stock_data)):
        month_list.append(int(dt.strptime( month_data.ix[i,"Date"], '%Y-%m-%d').month))

        """
        Code = int(code_data.ix[i, "code"])
        if 1300<=Code and Code<1500: code_list.append("1") # 農林水産

        elif 1500<=Code and Code<1700: code_list.append("2") # 鉱業

        elif 1700<=Code and Code<2000: code_list.append("3") # 建築

        elif 2000<=Code and Code<3000: code_list.append("4") # 食品

        elif 3000<=Code and Code<3600: code_list.append("5") # 繊維製品

        elif 3700<=Code and Code<4000: code_list.append("6") # パルプ・紙

        elif 4000<=Code and Code<5000: code_list.append("7") # 化学・医薬品

        elif 5000<=Code and Code<5100: code_list.append("8") # 石炭・石油

        elif 5100<=Code and Code<5200: code_list.append("9") # ゴム

        elif 5200<=Code and Code<5400: code_list.append("10") # 窯業

        elif 5400<=Code and Code<5700: code_list.append("11") # 鉄鋼

        elif 5700<=Code and Code<=5800: code_list.append("12") # 非金属

        elif 5900<=Code and Code<6000: code_list.append("13") # 金属製品

        elif 6000<=Code and Code<6500: code_list.append("14") # 機会

        elif 6500<=Code and Code<7000: code_list.append("15") # 電機

        elif 7000<=Code and Code<7500: code_list.append("16") # 輸送用機械

        elif 7700<=Code and Code<7800: code_list.append("17") # 精密機械

        elif 7800<=Code and Code<8000: code_list.append("18") # その他製品

        elif 8000<=Code and Code<8300: code_list.append("19") # 商業

        elif 8300<=Code and Code<8600: code_list.append("20") # 銀行・ノンバンク

        elif 8600<=Code and Code<8700: code_list.append("21") # 証券・証券先物

        elif 8700<=Code and Code<8800: code_list.append("22") # 保険

        elif 8800<=Code and Code<9000: code_list.append("23") # 不動産

        elif 9000<=Code and Code<9100: code_list.append("24") # 陸運

        elif 9100<=Code and Code<9200: code_list.append("25") # 海運

        elif 9200<=Code and Code<9300: code_list.append("26") # 空軍

        elif 9300<=Code and Code<9400: code_list.append("27") # 倉庫・運輸

        elif 9400<=Code and Code<9500: code_list.append("28") # 情報通信

        elif 9500<=Code and Code<9600: code_list.append("29") # 電機ガス

        elif 9600<=Code and Code<=9999: code_list.append("30") # サービス
        """
        code_list.append( str(code_data.ix[i, "code"])[0] )

    month_list_dummy = pd.DataFrame(month_list, columns=["month"])
    code_list_dummy = pd.DataFrame(code_list, columns=["class"])

    month_list_dummy = pd.get_dummies(month_list_dummy["month"])
    code_list_dummy = pd.get_dummies(code_list_dummy["class"])

    new_label_month = []
    for label in list(month_list_dummy.columns):
        new_label_month.append("month_"+str(label))

    new_label_class = []
    for label in list(code_list_dummy.columns):
        new_label_class.append("class_"+str(label))

    month_list_dummy.columns = new_label_month
    code_list_dummy.columns = new_label_class

    month_code_dummy = pd.concat([month_list_dummy, code_list_dummy], axis=1)

    return month_code_dummy



def main():
    variable = ["Date", "code", "Close"]
    data_path = "get_make_data/stock_value_data/"
    file_name = "2014-01-01_2015-01-01.csv"

    all_stock_data = data_load(variable, data_path + file_name)

    all_stock_data = Make_NextDayData_TeachData(all_stock_data, "Close")

    all_stock_data = get_SMA(all_stock_data, "Close")

    train_diff = get_train_diff(all_stock_data, "Close")

    #teach_diff = get_teach_diff(all_stock_data)
    teach_diff = get_teach_diff(all_stock_data, numeric=True)

    month_code_dummy = get_MonthCode_DummyData(all_stock_data)

    train = pd.concat([month_code_dummy, train_diff], axis=1)
    teach = teach_diff

    train.to_csv("get_make_data/x_t_data/x.csv", index=False)
    teach.to_csv("get_make_data/x_t_data/t.csv", index=False)
    return

if __name__ == '__main__':
    print("Granville_low_main")
    main()
