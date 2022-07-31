from datetime import datetime, timedelta
from utils import get_prices
import pandas as pd


if __name__ == "__main__":

    start_time = '1514764800'
    end_time = '1523404800'

    get_prices('btc', 'usdt',
               datetime(2019, 11, 27),
               duration=timedelta(days=69),
               proxy_port='8815',
               verbose=2)

    # df_ = pd.read_csv('data/BTCUSDT[' + start_time + '000_' + end_time + '000].csv')
    # df_.drop(['Unnamed: 0'], inplace=True, axis=1)
    # df_.set_index('Time', inplace=True, drop=True)
    #
    # times_index = pd.date_range(start=pd.to_datetime(start_time, unit='s'),
    #                             end=pd.to_datetime(end_time, unit='s'),
    #                             periods=int(end_time) - int(start_time) + 1)
    # df = pd.DataFrame(columns=df_.columns, index=times_index)
    # df.loc[df_.index] = df_
    #
    # df['Feature 1'] = df['Taker Buy Volume'] / df['Volume'] * 100
    # df['Feature 2'] = df['Volume'].pct_change()
    # df['Feature 3'] = df['Close'].pct_change()
    # tmp = df['High'].rolling(60).max()
    # df['Feature 4'] = (tmp-df['Open']) / (tmp-df['High'].rolling(60).min()) * 100
    #
    # print(df)
