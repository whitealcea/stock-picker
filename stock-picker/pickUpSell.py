import pandas as pd
from datetime import datetime, date, timedelta

df_id = pd.read_csv("./data/owned.csv")

sellStock = []

for num in range(len(df_id)):
    id = str(int(df_id.iloc[num]["id"]))
    print(id, "残り" + str(len(df_id) - num))

    try:
        df_stock = pd.read_csv("./data/" + id + ".csv")
    except FileNotFoundError: 
        print("skip:" + id)
        continue

    # MACDの計算を行う
    df_macd = pd.DataFrame()
    df_macd['date'] = df_stock['Date']
    df_macd['close'] = df_stock['Close']
    df_macd['ema_12'] = df_stock['Close'].ewm(span=12).mean()
    df_macd['ema_26'] = df_stock['Close'].ewm(span=26).mean()
    df_macd['macd'] = df_macd['ema_12'] - df_macd['ema_26']
    df_macd['signal'] = df_macd['macd'].ewm(span=9).mean()
    df_macd['histogram'] = df_macd['macd'] - df_macd['signal']
    df_macd.head()

    # ヒストグラムが凹んだ翌日に売る
    if df_macd.loc[len(df_macd)-1, "histogram"] > 0:
        if df_macd.loc[len(df_macd)-1, "histogram"] - df_macd.loc[len(df_macd)-2, "histogram"] < 0 :
            sellStock.append(int(id))

    df_macd.to_csv("./data/macd/" +id + ".csv")

print("=======================売りリスト=======================")
print(sellStock)