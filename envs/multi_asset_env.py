# Multi_asset_env module
class MultiAssetEnv:
    """
    Multi-Asset environment with short selling, partial fills, limit orders,
    advanced reward shaping, short borrow fees, dynamic position sizing constraints.

    Key expansions for "Optimal Execution":
      - Weighted position sizing logic
      - Short selling with fees
      - Multi-time-frame data (pass daily or intraday data in combined_df)
    """
    def __init__(self,
                 combined_df,
                 initial_investment=20000,
                 transaction_cost_pct=0.001,
                 slippage_pct=0.001,
                 stop_loss_pct=0.1,
                 risk_per_trade=0.1,
                 short_borrow_fee=0.0002,  # e.g., daily short fee
                 churn_penalty=0.0,
                 resolution='daily',
                 live_mode=False,
                 risk_window=10):
        # Sort data by date
        self.combined_df = combined_df.sort_index()
        self.resolution = resolution
        self.live_mode = live_mode

        # Partition columns: 1/3 price, 1/3 fund, 1/3 sentiment
        n_cols = self.combined_df.shape[1]
        self.price_cols = self.combined_df.columns[:n_cols//3]
        self.n_stock = len(self.price_cols)
        self.price_history = self.combined_df[self.price_cols].values
        self.fund_cols = self.combined_df.columns[n_cols//3: 2*n_cols//3]
        self.fundamentals = self.combined_df[self.fund_cols].values
        self.sent_cols = self.combined_df.columns[2*n_cols//3:]
        self.sentiments = self.combined_df[self.sent_cols].values
        self.dates = self.combined_df.index
        self.n_step = len(self.dates)

        # Execution placeholders
        # Average daily volume or similar. Large enough for partial fill logic
        self.adv = np.full(self.n_stock, 5e5)

        self.initial_investment = initial_investment
        self.cur_step = 0
        # Stock holdings can go negative => short selling
        self.stock_owned = np.zeros(self.n_stock, dtype=np.float32)
        self.stock_price = None
        self.cash_in_hand = float(initial_investment)

        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.stop_loss_pct = stop_loss_pct
        self.risk_per_trade = risk_per_trade
        self.short_borrow_fee = short_borrow_fee
        self.churn_penalty = churn_penalty

        # Discrete action space => 3^n_stock
        # 0 => Sell one "unit" (or partial), 1 => hold, 2 => buy one "unit"
        self.action_space = np.arange(3**self.n_stock)
        self.action_list = list(self._generate_actions())

        # Tech indicators
        self.rsi_data = np.zeros_like(self.price_history)
        self.ma_data = np.zeros_like(self.price_history)
        self.macd_data = np.zeros_like(self.price_history)
        self.macd_signal = np.zeros_like(self.price_history)
        self.bb_upper = np.zeros_like(self.price_history)
        self.bb_lower = np.zeros_like(self.price_history)

        for i in range(self.n_stock):
            arr = self.price_history[:, i]
            arr = wavelet_denoise(arr)
            self.rsi_data[:, i] = compute_rsi(arr)
            self.ma_data[:, i] = compute_moving_average(arr)
            macd, sig = compute_macd(arr)
            self.macd_data[:, i] = macd
            self.macd_signal[:, i] = sig
            up, lw = compute_bollinger(arr)
            self.bb_upper[:, i] = up
            self.bb_lower[:, i] = lw

        self.n_fund_feat = self.fundamentals.shape[1] // self.n_stock
        self.n_sent_feat = self.sentiments.shape[1] // self.n_stock

        # Each asset: [owned_shares, price, RSI, MA, MACD, MACD_sig, BB_up, BB_low, fundamentals..., sentiments...]
        # + 1 dimension for cash
        self.asset_state_dim = 8 + self.n_fund_feat + self.n_sent_feat
        self.state_dim = self.n_stock * self.asset_state_dim + 1

        # Reward tracking
        self.returns_history = deque(maxlen=risk_window)
        # For stop-loss reference
        self.buy_price = np.zeros(self.n_stock)
        # For portfolio and kill switch
        self.portfolio_vals = []
        self.max_val = initial_investment

        self.reset()

    def _generate_actions(self):
        """
        Return list of n_stock-digit ternary combos:
          e.g. for n_stock=2 => 3^2=9 combos => [ (0,0),(0,1),(0,2), (1,0),(1,1),...,(2,2) ]
        0 => SELL, 1 => HOLD, 2 => BUY
        """
        from itertools import product
        return product([0,1,2], repeat=self.n_stock)

    def reset(self):
        self.cur_step = 0
        self.stock_owned[:] = 0
        self.cash_in_hand = float(self.initial_investment)
        self.stock_price = self.price_history[self.cur_step]
        self.buy_price[:] = self.stock_price
        self.returns_history.clear()
        self.portfolio_vals = [self.initial_investment]
        self.max_val = self.initial_investment
        return self._get_obs()

    def step(self, action_idx):
        if self.live_mode:
            time.sleep(0.5)  # Emulate real-time feed

        prev_val = self._get_val()
        self.cur_step += 1
        if self.cur_step >= self.n_step:
            self.cur_step = self.n_step - 1
            done = True
        else:
            done = False

        self.stock_price = self.price_history[self.cur_step]

        # Execute trades
        self._trade(action_idx)
        self._stop_loss_check()

        # short borrow fees => charge if stock_owned[i]<0
        short_fee = self._apply_short_fee()

        cur_val = self._get_val()
        raw_pnl = cur_val - prev_val

        # Weighted position sizing or partial fill logic is inside _trade
        # Churn penalty if any non-HOLD
        action_vec = self.action_list[action_idx]
        n_moves = sum(a!=1 for a in action_vec)
        churn_cost = self.churn_penalty * n_moves

        # Overall return
        ret = raw_pnl / (prev_val + 1e-8)
        self.returns_history.append(ret)

        # Risk-based penalty
        vol = np.std(self.returns_history) if len(self.returns_history)>1 else 1.0
        var_ = compute_var(self.returns_history)
        cvar_ = compute_cvar(self.returns_history)

        # Reward = raw pnl adjusted by volatility + VaR + CVaR - churn cost - short fee
        reward = (raw_pnl / (vol + var_ + cvar_ + 1e-8)) - churn_cost - short_fee

        self.portfolio_vals.append(cur_val)
        self.max_val = max(self.max_val, cur_val)
        dd = (self.max_val - cur_val)/(self.max_val + 1e-8)
        # extra penalty for drawdown
        reward -= 0.01*dd

        # kill switch
        if kill_switch_check(self.portfolio_vals, max_dd=0.3):
            print("Kill-switch triggered due to excessive drawdown.")
            done = True

        if self.cur_step==self.n_step-1:
            done = True

        info = {'cur_val': cur_val, 'date': self.dates[self.cur_step]}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """
        Build state vector for RL agent.
        """
        obs = []
        for i in range(self.n_stock):
            chunk = [
                self.stock_owned[i],                # can be negative if short
                self.stock_price[i],
                self.rsi_data[self.cur_step, i],
                self.ma_data[self.cur_step, i],
                self.macd_data[self.cur_step, i],
                self.macd_signal[self.cur_step, i],
                self.bb_upper[self.cur_step, i],
                self.bb_lower[self.cur_step, i],
            ]
            if self.n_fund_feat>0:
                fstart = i*self.n_fund_feat
                fend = fstart+self.n_fund_feat
                chunk.extend(self.fundamentals[self.cur_step, fstart:fend])
            if self.n_sent_feat>0:
                sstart = i*self.n_sent_feat
                send = sstart+self.n_sent_feat
                chunk.extend(self.sentiments[self.cur_step, sstart:send])
            obs.extend(chunk)
        # last dim => cash in hand
        obs.append(self.cash_in_hand)
        return np.array(obs, dtype=np.float32)

    def _apply_short_fee(self):
        """
        If net short on any stock, charge daily short borrow fee proportional
        to # shares * price * short_borrow_fee.
        (Simplified approach.)
        """
        total_fee = 0.0
        for i in range(self.n_stock):
            if self.stock_owned[i]<0:  # short
                shares_short = abs(self.stock_owned[i])
                price = self.stock_price[i]
                fee = shares_short*price*self.short_borrow_fee
                if self.cash_in_hand>=fee:
                    self.cash_in_hand -= fee
                    total_fee+= fee
                else:
                    # forcibly reduce short if not enough cash to pay fees
                    # partial logic
                    feasible_shares = int(self.cash_in_hand/(price*self.short_borrow_fee+1e-8))
                    feasible_shares = min(feasible_shares, shares_short)
                    if feasible_shares>0:
                        self.cash_in_hand = 0
                        self.stock_owned[i]+= feasible_shares  # reduce short
                        total_fee+= feasible_shares*price*self.short_borrow_fee
        return total_fee

    def _simulate_slippage(self, shares):
        """
        Additional slippage factor for large orders relative to ADV.
        For demonstration, used in buy/sell calls below.
        """
        ratio = shares/1e5
        return self.slippage_pct*(1 + ratio*5)

    def _trade(self, action_idx):
        action_vec = self.action_list[action_idx]
        # Sell / Buy partial shares
        # Weighted position sizing => check 'risk_per_trade' logic
        # 0 => SELL 1 share, 1 => HOLD, 2 => BUY 1 share
        # We'll do SELL first, then BUY
        # short is allowed => if stock_owned < 0
        sell_indices = [i for i,a in enumerate(action_vec) if a==0]
        buy_indices = [i for i,a in enumerate(action_vec) if a==2]

        # SELL
        for i in sell_indices:
            # if we hold x>0 shares => reduce
            # if we hold x<0 => go further short
            self._execute_sell(i, 1, order_type="limit")

        # BUY
        for i in buy_indices:
            self._execute_buy(i, 1, order_type="limit")

    def _execute_buy(self, i, shares, order_type="market"):
        """
        Weighted position sizing: can only buy if not exceeding self.risk_per_trade * portfolio_value
        position_value + new_shares*price <= risk_per_trade * total_value
        """
        price = self.stock_price[i]
        limit_price = price*1.01 if order_type=="limit" else None
        if order_type=="limit" and price>limit_price:
            # no fill
            return 0
        # slippage + fee
        slip_pct = self._simulate_slippage(shares)
        cost_no_fee = price*shares
        fee = cost_no_fee*self.transaction_cost_pct
        slip = cost_no_fee*slip_pct
        total_cost = cost_no_fee + fee + slip

        # risk-based position sizing
        # current_position_val = abs(self.stock_owned[i])*price
        # max_allowed = self._get_val() * self.risk_per_trade
        # if current_position_val + cost_no_fee> max_allowed: # reduce shares
        #   ...
        # For simplicity, only do a check:
        max_allowed = self._get_val() * self.risk_per_trade
        current_val = abs(self.stock_owned[i])*price
        if current_val + cost_no_fee> max_allowed:
            # reduce shares
            feasible_shares = int((max_allowed - current_val)/(price+1e-8))
            feasible_shares = max(0, feasible_shares)
            if feasible_shares<shares:
                shares = feasible_shares
                total_cost = shares*price + shares*price*(self.transaction_cost_pct+slip_pct)

        if shares<=0:
            return 0

        if self.cash_in_hand>=total_cost:
            self.cash_in_hand-= total_cost
            self.stock_owned[i]+= shares
            if self.stock_owned[i]>0:
                self.buy_price[i] = price  # reset buy price for stop-loss
            log_trade(f"BUY {shares} shares of {i} @ {price:.2f}, cost={total_cost:.2f}")
            return shares
        return 0

    def _execute_sell(self, i, shares, order_type="market"):
        """
        If we have x>0 => reduce x (long->less long).
        If x<0 => more short.
        """
        price = self.stock_price[i]
        limit_price = price*0.99 if order_type=="limit" else None
        if order_type=="limit" and price<limit_price:
            return 0

        slip_pct = self._simulate_slippage(shares)
        proceeds_no_fee = price*shares
        fee = proceeds_no_fee*self.transaction_cost_pct
        slip = proceeds_no_fee*slip_pct
        net_proceeds = proceeds_no_fee - fee - slip

        # Check short risk => if we are going from x<0 to more negative
        # Possibly allow infinite short for demo, or do risk check:
        # skip advanced logic for brevity

        if self.stock_owned[i]>=0:
            # normal sell of long position
            if self.stock_owned[i]<shares:
                shares = self.stock_owned[i]
            self.stock_owned[i]-= shares
            self.cash_in_hand+= net_proceeds
        else:
            # further short
            self.stock_owned[i]-= shares
            self.cash_in_hand+= net_proceeds
        if self.stock_owned[i]<0:
            # do not change buy_price for short
            pass
        else:
            # if we closed or reduced a long
            pass

        log_trade(f"SELL {shares} shares of {i} @ {price:.2f}, proceeds={net_proceeds:.2f}")
        return shares

    def _stop_loss_check(self):
        for i in range(self.n_stock):
            if self.stock_owned[i]>0:
                trigger = self.buy_price[i]*(1-self.stop_loss_pct)
                if self.stock_price[i]<trigger:
                    # sell all
                    self._execute_sell(i, self.stock_owned[i])

    def _get_val(self):
        # Owned can be negative => short
        # Value of negative is "borrowed" => but netVal = cash + sum(owned[i]*price[i])
        total_val = self.cash_in_hand + np.sum(self.stock_owned*self.stock_price)
        return total_val

