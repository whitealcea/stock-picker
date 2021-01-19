import urllib3
from bs4 import BeautifulSoup
import time
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import pandas_datareader.data as web
import requests
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

baseUrl = "https://minkabu.jp/stock/"

lastDay = datetime.now().date()

upperStock = []
buyTimingStock = []

df_id = pd.read_csv("./resources/company.csv")
http = urllib3.PoolManager()

def up75d(df):
    """
    75日線が上向きか
    """
    # 直近3日
    # return df.loc[len(df)-1, "Ma_75d_r1"] > 0 and df.loc[len(df)-1, "Ma_75d_r2"] > 0 and df.loc[len(df)-1, "Ma_75d_r3"] > 0
    # 直近1日
    return df.loc[len(df)-1, "Ma_75d_r1"] > 0

def up5d(df):
    """
    5日線が上向きか
    """
    return df.loc[len(df)-1, "Ma_5d_r1"] > 0 and df.loc[len(df)-1, "Ma_5d_r2"] > 0 and df.loc[len(df)-1, "Ma_5d_r3"] > 0

for num in range(len(df_id)):
    id = str(int(df_id.iloc[num]["id"]))

    url = baseUrl + id
    r = http.request('GET',url)
    soup = BeautifulSoup(r.data, "html.parser")
    targetList = soup.findAll(True, {'class':['ly_row ly_gutters']})[1].findAll(True, {'class':['md_list']})[0].findAll(True, {'class':['ly_vamd_inner ly_colsize_9_fix fwb tar wsnw']})
    endHtml = soup.find(id="stock-for-securities-company")
    print(id, "残り" + str(len(df_id) - num))
    url = baseUrl + id
    r = http.request('GET',url)
    soup = BeautifulSoup(r.data, "html.parser")
    if targetList[0].getText() == '---':
        continue
    open = float(targetList[0].getText().replace(",", "").replace("円", ""))
    high = float(targetList[1].getText().replace(",", "").replace("円", ""))
    low = float(targetList[2].getText().replace(",", "").replace("円", ""))
    close = float(soup.findAll(True, {'class':['stock_price']})[0].getText().replace(" ", "").replace(",", "").replace("円", ""))

    volume = int(soup.findAll(True, {'class':['ly_row ly_gutters']})[1].findAll(True, {'class':['md_list']})[2].findAll(True, {'class':['ly_vamd_inner ly_colsize_9_fix fwb tar wsnw']})[0].getText().replace(",", "").replace("株", ""))
    df = web.DataReader(id + '.JP', 'stooq')

    df = df.iloc[::-1]
    df['Date'] = df.index

    tmp_se = pd.Series( [ open, low, high, close, volume, lastDay ], index=df.columns )
    df = df.append(tmp_se, ignore_index=True)
    print(id, "始値：", open, "高値：", high, "安値：", low, "終値：", close, "出来高：", volume)
    num = num + 1

    df['Ma_5d'] = df['Close'].rolling(window=5).mean()
    df['Ma_25d'] = df['Close'].rolling(window=25).mean()
    df['Ma_50d'] = df['Close'].rolling(window=50).mean()
    df['Ma_75d'] = df['Close'].rolling(window=75).mean()
    df['Ma_100d'] = df['Close'].rolling(window=100).mean()
    df['Ma_150d'] = df['Close'].rolling(window=150).mean()
    df['Ma_200d'] = df['Close'].rolling(window=200).mean()

    df['Result'] = ""

    for num2 in range(1, len(df)):
        df.loc[num2, 'Open_tdb'] = df.loc[num2, 'Open'] - df.loc[num2 - 1, 'Open']
        df.loc[num2, 'Low_tdb'] = df.loc[num2, 'Low'] - df.loc[num2 - 1, 'Low']
        df.loc[num2, 'High_tdb'] = df.loc[num2, 'High'] - df.loc[num2 - 1, 'High']
        df.loc[num2, 'Close_tdb'] = df.loc[num2, 'Close'] - df.loc[num2 - 1, 'Close']
        df.loc[num2, 'Volume_tdb'] = df.loc[num2, 'Volume'] - df.loc[num2 - 1, 'Volume']
        df.loc[num2, 'Tdb_rate'] = df.loc[num2, 'Close_tdb'] / df.loc[num2 - 1, 'Close']
        df.loc[num2, "Ma_5d_r1"] = (df.loc[num2, "Ma_5d"] - df.loc[num2 - 1, "Ma_5d"]) / df.loc[num2 - 1, "Ma_5d"]
        df.loc[num2, "Ma_25d_r1"] = (df.loc[num2, "Ma_25d"] - df.loc[num2 - 1, "Ma_25d"]) / df.loc[num2 - 1, "Ma_25d"]
        df.loc[num2, "Ma_75d_r1"] = (df.loc[num2, "Ma_75d"] - df.loc[num2 - 1, "Ma_75d"]) / df.loc[num2 - 1, "Ma_75d"]
        df.loc[num2, "Ma_5d_r2"] = df.loc[num2 - 1, "Ma_5d_r1"]
        df.loc[num2, "Ma_25d_r2"] = df.loc[num2 - 1, "Ma_25d_r1"]
        df.loc[num2, "Ma_75d_r2"] = df.loc[num2 - 1, "Ma_75d_r1"]
        df.loc[num2, "Ma_5d_r3"] = df.loc[num2 - 1, "Ma_5d_r2"]
        df.loc[num2, "Ma_25d_r3"] = df.loc[num2 - 1, "Ma_25d_r2"]
        df.loc[num2, "Ma_75d_r3"] = df.loc[num2 - 1, "Ma_75d_r2"]
        resultNum = 0
        if df.loc[num2, 'Close'] >= df.loc[num2 - 1, 'Close']:
            resultNum = 1
        df.loc[num2 - 1, 'Result'] = resultNum

    # 半年以内で3ヶ月下げ続けたか
    #  → 1週間以前の最高値が2,3ヶ月前であること
    halfYdf = df[df["Date"] >= (datetime.now().date() - relativedelta(months=6))]

    if(df.loc[halfYdf["High"].idxmax(), "Date"] < (datetime.now().date() - relativedelta(months=2))):
        # 75日線が上向きか
        if up75d(df):
            # 株価が上昇していること
            if df.loc[len(df)-1, "Open"] >= df.loc[len(df)-2, "Open"]:
                # 株価が75日線の上に出たタイミング
                Ma_75d = df.loc[len(df)-1, "Ma_75d"]
                if  -(Ma_75d // 20) <= df.loc[len(df)-1, "High"] - Ma_75d <= (Ma_75d // 20):
                    buyTimingStock.append(id)
                else:
                    upperStock.append(id)
    df.to_csv("./data/" + str(id) + ".csv")

print("=======================買いリスト=======================")
print("　上昇中：")
print(upperStock)
print("　買いタイミング：")
print(buyTimingStock)

df_result = pd.read_csv("./data/result.csv", index_col=0)
size = len(df_result)
df_result.loc[size, "date"] = lastDay
df_result.loc[size, "upperStock"] = str(upperStock)
df_result.loc[size, "buyTimingStock"] = str(buyTimingStock)
df_result.to_csv("./data/result.csv")