from datetime import datetime, timedelta
from utils import get_prices
import pandas as pd


if __name__ == "__main__":
    # get_prices('btc', 'usdt',
    #            datetime(2018, 1, 1),
    #            duration=timedelta(days=2),
    #            proxy_port='59742',
    #            verbose=2)
    df = pd.read_csv('data/BTCUSDT[1514764800000_1514937600000].csv')
    df.drop(['Unnamed: 0'], inplace=True, axis=1)

    df['Feature 1'] = df['Taker Buy Volume'] / df['Volume'] * 100
    df['Feature 2'] = df['Volume'].pct_change()
    print(df)


