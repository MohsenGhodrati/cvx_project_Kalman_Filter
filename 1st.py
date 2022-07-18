import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
from pathlib import Path


ROOT = Path().resolve()


def get_prices(coin):
    milliseconds = 1000
    start_time = int(datetime(2018, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()) * milliseconds
    end_time = int(datetime(2018, 1, 3, 0, 0, 0, tzinfo=timezone.utc).timestamp()) * milliseconds

    df = list()

    params = {
        'symbol': coin + 'USDT',
        'interval': '1m',
        'limit': 1000,
        'startTime': 0,
        'endTime': 0
    }

    time_iterator = start_time
    while time_iterator < end_time:
        params['startTime'] = time_iterator
        params['endTime'] = min(time_iterator + params['limit'] * milliseconds * 60, end_time) - 1
        response = requests.get(
            url="https://api.binance.com/api/v3/klines",
            proxies={"https": "http://localhost:61041"},
            params=params
        )

        result = list(response.json())
        df.extend(result)
        time_iterator += params['limit'] * milliseconds * 60

    df = pd.DataFrame(df, columns=[
        'Time',
        'Open',
        'High',
        'Low',
        'Close',
        'Volume',
        'Close Time',
        'Value',
        'No. Trades',
        'Taker Buy Volume',
        'Taker Buy Value',
        'Ignore'])

    df = df.astype({
        'Open': float,
        'High': float,
        'Low': float,
        'Close': float,
        'Volume': float,
        'Value': float,
        'No. Trades': int,
        'Taker Buy Volume': float,
        'Taker Buy Value': float,
        'Ignore': float
    })

    #     if all(df['Close Time'] - df['Time'] == 59999):
    df.drop(['Close Time', 'Ignore'], inplace=True, axis=1)
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')

    df.insert(0, 'Market', params['symbol'])

    df.to_csv(str(ROOT) + '/data/' + coin + 'USDT.csv')
    return df


get_prices('BTC')

