# -*- coding: utf_8 -*-  

import numpy as np
import pandas as pd
import pandas.io.data as web
import sys

def get_quote_yahoojp(code, start=None, end=None, interval='d'):
    base = 'http://info.finance.yahoo.co.jp/history/?code={0}.T&{1}&{2}&tm={3}&p={4}'
    start, end = web._sanitize_dates(start, end)
    start = 'sy={0}&sm={1}&sd={2}'.format(start.year, start.month, start.day)
    end = 'ey={0}&em={1}&ed={2}'.format(end.year, end.month, end.day)
    p = 1 #ページ
    results = []

    if interval not in ['d', 'w', 'm', 'v']:
        raise ValueError("Invalid interval: valid values are 'd', 'w', 'm' and 'v'")

    while True:
        url = base.format(code, start, end, interval, p)
        try:
            tables = pd.read_html(url, header=0)

        except ValueError:
            #print("Value Error")
            return []

        if len(tables) < 2 or len(tables[1]) == 0:
            break
        results.append(tables[1])
        p += 1

    if len(results)==0:
        return []
    else:
        result = pd.concat(results, ignore_index=True)

    result.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    if interval == 'm':
        result['Date'] = pd.to_datetime(result['Date'], format='%Y年%m月')
    else:
        result['Date'] = pd.to_datetime(result['Date'], format='%Y年%m月%d日')

    result = pd.concat([result,pd.DataFrame({"code":[code]*len(result)})],axis=1)
    result = result.set_index('Date')
    result = result.sort_index()

    date = pd.DataFrame(result.index)
    result.index = range(len(result))
    result = pd.concat([date, result],axis=1)
    return result

if __name__ == "__main__":
    code_nums = pd.read_csv("tosyo1.csv")
    #code_nums = pd.DataFrame([1432])
    start = "2014-01-01"
    end = "2015-01-01"

    output_path = "stock_value_data/"
    all_result = None

    for i in range(len(code_nums)):
        code = code_nums.ix[i,0]
        print(code)
        result = get_quote_yahoojp(code, start, end)
        if len(result)==0:
            continue
        else:
            #result.to_csv(output_path+str(code)+"_"+start+"_"+end+".csv")
            all_result = pd.concat([all_result, result], axis=0)
        #sys.exit()
    all_result.index = range(len(all_result))
    print(all_result)

    all_result.to_csv(output_path+start+"_"+end+".csv")
