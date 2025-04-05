def load_price_data():
    """
    Loads daily price data from 'aapl_msi_sbux.csv' located in your prices folder.
    CSV Format: date, AAPL, MSI, SBUX, ...
    """
    file_path = r"C:\Users\abdou\OneDrive\Bureau\prices\aapl_msi_sbux.csv"
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    df.sort_index(inplace=True)
    return df

def load_intraday_data(resolution='1min'):
    """
    Tries to load intraday data from e.g. 'intraday_prices_1min.csv'.
    Otherwise fall back to daily data.
    """
    try:
        df = pd.read_csv(f'intraday_prices_{resolution}.csv', parse_dates=['date'], index_col='date')
        df.sort_index(inplace=True)
        return df
    except Exception:
        print(f"No intraday data for {resolution}, using daily fallback.")
        return load_price_data()

def load_fundamental_data():
    """
    Loads fundamentals from 'fundamentals.csv' if available.
    """
    try:
        df = pd.read_csv('fundamentals.csv', parse_dates=['date'], index_col='date')
        df.sort_index(inplace=True)
        return df
    except:
        print("No fundamentals found. Using zeros.")
        return None

def load_sentiment_data():
    """
    Loads sentiment from 'sentiment.csv' if available.
    Otherwise create zero placeholders matching price data shape.
    """
    try:
        df = pd.read_csv('sentiment.csv', parse_dates=['date'], index_col='date')
        df.sort_index(inplace=True)
        return df
    except:
        print("No sentiment data found, using zeros.")
        price_df = load_price_data()
        df = pd.DataFrame(
            np.zeros((len(price_df), price_df.shape[1])),
            index=price_df.index,
            columns=[f"{c}_sent" for c in price_df.columns]
        )
        return df

def merge_data(price_df, fund_df, sent_df):
    """
    Merge price, fundamentals, sentiment into a single DataFrame (combined_df).
    We assume:
      - price_df columns: [AAPL, MSI, SBUX, ...]
      - fundamentals => appended as _fund
      - sentiment => appended as _sent
    """
    if fund_df is None:
        fund_df = pd.DataFrame(
            np.zeros((len(price_df), price_df.shape[1])),
            index=price_df.index,
            columns=[f"{c}_fund" for c in price_df.columns]
        )
    else:
        fund_df = fund_df.reindex(price_df.index).fillna(method='ffill').fillna(0)

    if sent_df is None:
        sent_df = pd.DataFrame(
            np.zeros((len(price_df), price_df.shape[1])),
            index=price_df.index,
            columns=[f"{c}_sent" for c in price_df.columns]
        )
    else:
        sent_df = sent_df.reindex(price_df.index).fillna(0)

    combined = pd.concat([price_df, fund_df, sent_df], axis=1)
    return combined
