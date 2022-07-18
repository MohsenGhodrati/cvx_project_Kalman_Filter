import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
from contexttimer import Timer
from __init__ import ROOT


def get_prices(
        base_coin: str,
        quote: str,
        start_time: datetime,
        duration: timedelta = timedelta(days=1),
        interval: timedelta = timedelta(minutes=1),
        proxy_port: str = None,
        verbose: int = 0
):

    milliseconds = 1000
    start_datetime = datetime(
        start_time.year, start_time.month, start_time.day,
        start_time.hour, start_time.minute, start_time.second,
        tzinfo=timezone.utc)
    start_time_ = int(start_datetime.timestamp()) * milliseconds
    end_datetime = start_datetime + duration
    end_time_ = int(end_datetime.timestamp()) * milliseconds

    df = list()

    interval_ = '1m'
    if int(interval / timedelta(hours=1)) == 1:
        interval_ = '1h'
    if int(interval / timedelta(days=1)) == 1:
        interval_ = '1d'

    params = {
        'symbol': base_coin.upper() + quote.upper(),
        'interval': interval_,
        'limit': 1000,
        'startTime': 0,
        'endTime': 0
    }

    if proxy_port is None:
        proxy = None
    else:
        proxy = {
            "https": "http://localhost:" + proxy_port
        }

    time_iterator = start_time_
    counter = 0
    while time_iterator < end_time_:
        params['startTime'] = time_iterator
        params['endTime'] = \
            min(time_iterator + params['limit'] * milliseconds * int(interval.seconds), end_time_) - 1

        counter += 1
        with Timer(output=verbose >= 2, fmt="Request " + str(counter) + " took {:.3f} seconds..."):
            response = requests.get(
                url="https://api.binance.com/api/v3/klines",
                proxies=proxy,
                params=params
            )

        result = list(response.json())
        df.extend(result)
        time_iterator += params['limit'] * milliseconds * int(interval.seconds)

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

    if all(df['Close Time'] - df['Time'] == int(interval.seconds) * milliseconds - 1):
        df.drop(['Close Time', 'Ignore'], inplace=True, axis=1)
    else:
        if verbose >= 1:
            print("Missing candles...")
        df.drop(['Close Time', 'Ignore'], inplace=True, axis=1)
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    df.insert(0, 'Market', params['symbol'])

    if len(df) == 100:
        if verbose >= 1:
            print("{:s}: complete data received.".format(start_time))
    df.to_csv(str(ROOT) + '/data/' + base_coin.upper() + quote.upper() +
              '[' + str(start_time_) + '_' + str(end_time_) + ']' + '.csv')
    return df
