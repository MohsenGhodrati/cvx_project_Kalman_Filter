import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
from contexttimer import Timer
import matplotlib.pyplot as plt
import seaborn as sns
from __init__ import ROOT, EPSILON, RETURN_SCALE


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


def plot_heatmap(coeffs, title):
    plt.figure(figsize=(20, 5), dpi=300)
    sns.heatmap(coeffs.T, xticklabels=288, center=0)
    plt.xticks(rotation=45, fontsize=8)
    ticks = plt.xticks()
    plt.xticks(ticks[0], ['{:s}'.format(i.get_text()[2:10]) for i in ticks[1]])
    plt.xlabel('time')
    plt.ylabel('relative past')
    plt.suptitle('Heatmap for Matrix Coefficients')
    plt.title(title, fontsize=10)
    plt.savefig(str(ROOT) + f'/plots/Heatmap [{title}].png')
    plt.show()


def plot_errors_hist(results, title, limits=None):
    if limits is None:
        limits = {'x': [-0.006, 0.006], 'y': [0, 500]}
    tmp = results['Measurement'] - results['Reality']
    plt.hist(
        tmp,
        bins=1000, alpha=0.5, color='purple', density=True,
        label=r'$Measurement\ (\mu={:.2f}, \sigma^2={:.2f})$'.format(tmp.mean()*1000000, tmp.var()*1000000))

    tmp = results['Prediction'] - results['Reality']
    plt.hist(
        tmp,
        bins=1000, alpha=0.5, color='green', density=True,
        label=r'$Model\ (\mu={:.2f}, \sigma^2={:.2f})$'.format(tmp.mean()*1000000, tmp.var()*1000000))

    plt.xlim(limits['x'])
    plt.legend(title='Scaled to million units', fontsize=8)
    plt.suptitle('Estimation Errors Histogram')
    plt.title(title, fontsize=10)
    plt.xlabel('error')
    plt.ylabel('density')
    plt.ylim(limits['y'])
    plt.savefig(str(ROOT) + f'/plots/Errors Histogram [{title}].png', dpi=300)
    plt.show()


def plot_estimators_hist(results, title, limits=None):

    if limits is None:
        limits = {'x': [-0.0075, 0.0075], 'y': [0, 1800]}

    results['Reality'].hist(
        bins=300, alpha=0.5, density=True,
        label=r'$Reality\ (\mu={:.2f}, \sigma^2={:.2f})$'.format(
            results['Reality'].mean()*1000000, results['Reality'].var()*1000000))
    results['Measurement'].hist(
        bins=300, alpha=0.5, color='green', density=True,
        label=r'$Measurement\ (\mu={:.2f}, \sigma^2={:.2f})$'.format(
            results['Measurement'].mean()*1000000, results['Measurement'].var()*1000000))
    results['Prediction'].hist(
        bins=300, alpha=0.5, color='red', density=True,
        label=r'$Model\ (\mu={:.2f}, \sigma^2={:.2f})$'.format(
            results['Prediction'].mean()*1000000, results['Prediction'].var()*1000000))

    plt.xlim(limits['x'])
    plt.legend(title='Scaled to million units', fontsize=8)
    plt.suptitle('Distribution of Estimators vs. Reality')
    plt.title(title, fontsize=10)
    plt.xlabel('next return (price change)')
    plt.ylabel('frequency')
    plt.ylim(limits['y'])
    plt.savefig(str(ROOT) + f'/plots/Estimators Histogram [{title}].png', dpi=300)
    plt.show()


def plot_returns_and_filter(results, indices, title, colors=None):

    if colors is None:
        colors = {
            'Estimate': 'red',
            'Measurement': 'blue',
            'Prediction': 'skyblue'
        }

    plt.figure(figsize=(20, 8), dpi=60)

    plt.xticks(rotation=90)
    plt.plot(
        indices,
        results.loc[indices, 'Reality'],
        color='black',
        label='Real Value'
    )

    # plot measurement errors
    plt.vlines(
        x=indices,
        ymin=results.loc[indices, 'Reality'],
        ymax=results.loc[indices, 'Measurement'],
        color=colors['Measurement'],
        alpha=0.2
    )
    plt.scatter(
        indices,
        results.loc[indices, 'Measurement'],
        color=colors['Measurement'],
        label='Measurement'
    )

    # plot model approximation errors
    plt.vlines(
        x=indices,
        ymin=results.loc[indices, 'Reality'],
        ymax=results.loc[indices, 'Prediction'],
        color=colors['Prediction'],
        alpha=0.2
    )
    plt.scatter(
        indices,
        results.loc[indices, 'Prediction'],
        color=colors['Prediction'],
        label='Model Estimate'
    )

    # plot model approximation errors
    # plt.vlines(
    #     x=indices,
    #     ymin=results.loc[indices, 'Reality'],
    #     ymax=results.loc[indices, 'Estimate'],
    #     color=colors['Estimate'],
    #     alpha=.2
    # )
    plt.scatter(
        indices,
        results.loc[indices, 'Estimate'],
        color=colors['Estimate'],
        marker='x',
        label='Kalman Estimate'
    )
    plt.plot(
        indices,
        results.loc[indices, 'Estimate'],
        color='black',
        linestyle=':'
    )

    plt.suptitle('Comparison of Kalman Filter Estimates with Initial Ones')
    plt.title(title, fontsize=10)
    plt.ylabel('return (price change)')
    plt.xlabel('time')
    plt.legend()

    plt.fill_between(
        indices,
        results.loc[indices, 'Estimate'],
        results.loc[indices, 'Reality'],
        color=colors['Estimate'],
        alpha=0.3
    )

    plt.savefig(str(ROOT) + f'/plots/Returns and Kalman Filtering [{title}].png')
    plt.show()


def load_results(title, k, m, alpha_inv=None):

    if title == 'Linear Model without Control':
        filename = f'linear_model_without_control[k={k},m={m}]'
    elif title == 'Linear Model with Control':
        filename = f'linear_model_with_control[k={k},m={m}]'
    elif title.startswith('Weighted (') and title.endswith(') Linear Model with Control'):
        if alpha_inv is not None:
            alpha = 1. / (alpha_inv / 100)
        filename = f'weighted_linear_model_with_control[k={k},m={m},a={alpha}]'

    results = pd.read_csv(str(ROOT) + f'/data/{filename}')
    results['Unnamed: 0'] = pd.to_datetime(results['Unnamed: 0'])
    results.set_index('Unnamed: 0', drop=True, inplace=True)
    results.index.name = None
    results = results.astype(float)
    return results


def apply_kalman_filter(results, std_windows=None):
    if std_windows is None:
        std_windows = {'Measurement': 12 * 6, 'Prediction': 12}

    results['Measurement Residual'] = results['Measurement'] - results['Reality']
    results['Prediction Residual'] = results['Prediction'] - results['Reality']

    a = results['Measurement Residual'].rolling(std_windows['Measurement']).var()
    results['Measurement Residual Weighted Variance'] = a

    b = results['Prediction Residual'].rolling(std_windows['Prediction']).var()
    results['Prediction Residual Weighted Variance'] = b

    results['Kalman Gain'] = b / (a + b)
    results['Estimate'] = results['Kalman Gain'] * results['Measurement'] + \
                          (1 - results['Kalman Gain']) * results['Prediction']

    results['Estimate Uncertainty'] = results['Kalman Gain'] ** 2 * a + \
                                      (1 - results['Kalman Gain']) ** 2 * b
    return results


def compute_features(df):
    features = pd.DataFrame(index=df.index)
    features['taker buy volume ratio'] = df['Taker Buy Volume'] / (df['Volume'] + EPSILON)
    features['volume (1e3)'] = df['Volume'] / 1000
    features['volume change'] = (df['Volume'] + EPSILON).pct_change()
    features['trades number (1e3)'] = df['No. Trades'] / 1000
    features['trades number change'] = (df['No. Trades'] + EPSILON).pct_change()
    features['taker buy average spread (1e-3)'] = \
        ((df['Taker Buy Value'] / df['Taker Buy Volume']) / df['Average Price'] - 1) * 1000
    features['taker sell average spread (1e-3)'] = \
        (((df['Value'] - df['Taker Buy Value']) /
          (df['Volume'] - df['Taker Buy Volume'])) / df['Average Price'] - 1) * 1000
    # def historical_high_to_low_ratio(window):
    #     tmp = df['High'].rolling(window).max()
    #     return (tmp-df['Close']) / (tmp-df['Low'].rolling(window).min() + EPSILON) * 100
    #
    # for i, t in enumerate([1, 6, 24]):
    #     window = int(timedelta(hours=t) / timeframe)
    #     df[f'Feature {4+i}'] = historical_high_to_low_ratio(window)

    # features_means, features_stds = features.mean(), features.std()
    # features = (features - features_means) / features_stds
    # print('Multipliers:')
    # print(features_stds)
    return features


def resample_data(df_, timeframe, start_time_, end_time_):
    start_time = int(start_time_.timestamp())
    end_time = int(end_time_.timestamp())
    times_index = pd.date_range(
        start=pd.to_datetime(start_time, unit='s'),
        end=pd.to_datetime(end_time, unit='s'),
        periods=(int(end_time) - int(start_time) + 1) / 60 + 1)
    df = pd.DataFrame(None, columns=df_.columns, index=times_index[:-1])
    selected_indices = list(set(df_.index).intersection(set(times_index)))
    df.loc[selected_indices, df_.columns] = df_.loc[selected_indices].copy()

    df = df.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Value': 'sum',
        'No. Trades': 'sum',
        'Taker Buy Volume': 'sum',
        'Taker Buy Value': 'sum'
    })
    df['Average Price'] = df['Value'] / df['Volume']
    df['Average Price Change'] = df['Average Price'].pct_change()

    return df


def load_data(start_time_, end_time_):
    start_time = int(start_time_.timestamp())
    end_time = int(end_time_.timestamp())
    df_ = pd.read_csv('data/BTCUSDT[' + str(start_time) + '000_' + str(end_time) + '000].csv')
    df_.drop(['Unnamed: 0'], inplace=True, axis=1)
    df_['Time'] = pd.to_datetime(df_['Time'])
    df_.set_index('Time', inplace=True, drop=True)
    return df_