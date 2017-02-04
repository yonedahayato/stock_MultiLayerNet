# coding: utf-8

from urllib.request import *
from lxml import html
import numpy as np
import pandas as pd

import sys

#東証１部の銘柄一覧
link = "http://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

def getStockNameDF():
    f = "http://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    df = pd.ExcelFile(f).parse('Sheet1')

    return pd.DataFrame({'code': df[u"コード"].astype('int64')})

def saveCSV(df):
    df[['code']].to_csv('tosyo1.csv', index=False, encoding='utf-8')

if __name__ == '__main__':
    df = getStockNameDF()
    saveCSV(df)
