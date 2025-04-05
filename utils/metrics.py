# Metrics module
def compute_var(returns, confidence=0.95):
    if len(returns)==0:
        return 0
    sorted_r = np.sort(returns)
    idx = int((1-confidence)*len(sorted_r))
    return abs(sorted_r[idx])

def compute_cvar(returns, confidence=0.95):
    if len(returns)==0:
        return 0
    sorted_r = np.sort(returns)
    idx = int((1-confidence)*len(sorted_r))
    tail = sorted_r[:idx+1]
    return abs(np.mean(tail))

def kill_switch_check(port_vals, max_dd=0.2):
    """
    If portfolio drawdown > max_dd => kill switch triggered.
    """
    if not port_vals:
        return False
    peak = max(port_vals)
    dd = (peak - port_vals[-1]) / (peak+1e-8)
    return dd > max_dd

def log_trade(msg, logfile='trade_log.txt'):
    with open(logfile, 'a') as f:
        f.write(f"{datetime.now()} => {msg}\n")