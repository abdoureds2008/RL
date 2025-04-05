def wavelet_denoise(arr):
    """
    If pywt is installed, do a wavelet-based denoising.
    Otherwise, return original arr.
    """
    if pywt is None:
        return arr
    coeffs = pywt.wavedec(arr, 'db1', level=2)
    thresh = np.median(np.abs(coeffs[-1])) * 0.5
    coeffs = [np.where(np.abs(c)<thresh, 0, c) for c in coeffs]
    rec = pywt.waverec(coeffs, 'db1')
    return rec[:len(arr)]

def compute_rsi(prices, window=14):
    deltas = np.diff(prices)
    rsi = np.zeros_like(prices)
    up = np.where(deltas>0, deltas, 0)
    down = np.where(deltas<0, -deltas, 0)
    alpha = 1.0/window
    up_ema, down_ema = 0, 0
    for i in range(1, len(prices)):
        up_ema = alpha*up[i-1] + (1-alpha)*up_ema
        down_ema = alpha*down[i-1] + (1-alpha)*down_ema
        rs = up_ema / down_ema if down_ema else 0
        rsi[i] = 100 - (100/(1+rs))
    return rsi

def compute_moving_average(prices, window=5):
    return np.convolve(prices, np.ones(window), 'same') / window

def compute_macd(prices, short=12, long=26, signal=9):
    s = pd.Series(prices)
    exp1 = s.ewm(span=short, adjust=False).mean()
    exp2 = s.ewm(span=long, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line.values, signal_line.values

def compute_bollinger(prices, window=20, num_std=2):
    s = pd.Series(prices)
    ma = s.rolling(window).mean()
    std = s.rolling(window).std()
    upper = ma + num_std*std
    lower = ma - num_std*std
    return upper.values, lower.values




