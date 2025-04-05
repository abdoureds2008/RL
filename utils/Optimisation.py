def hyperparam_optimization(trial):
    lr_ = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    gamma_ = trial.suggest_uniform('gamma', 0.9, 0.99)
    epsd_ = trial.suggest_uniform('eps_decay', 0.990, 0.999)

    price_df = load_price_data()
    fund_df = load_fundamental_data()
    sent_df = load_sentiment_data()
    combo = merge_data(price_df, fund_df, sent_df)
    half = len(combo)//2
    train_df = combo.iloc[:half]

    env = MultiAssetEnv(train_df)
    s_size = env.state_dim
    a_size = len(env.action_space)
    agent = DoubleDQNAgent(
        s_size, a_size,
        gamma=gamma_, epsilon_decay=epsd_, lr=lr_
    )

    sc = get_scaler(env, num_samples=500)
    total_val = 0
    episodes = 5
    for _ in range(episodes):
        v = play_one_episode(agent, env, sc, mode='train')
        total_val+= v
    return -total_val/episodes

