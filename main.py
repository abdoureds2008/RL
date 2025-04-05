from utils.imports_and_gpu import configure_gpu
configure_gpu()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train', 'test', 'walkforward', 'plot'],
                        help='train/test/walkforward/plot')
    parser.add_argument('--agent', default='discrete', choices=['discrete','continuous','ensemble'],
                        help='RL agent type')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay for discrete agent')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--resolution', default='daily', help='daily or 1min, etc.')
    parser.add_argument('--live', action='store_true', help='If set, slow down steps to emulate real-time.')
    parser.add_argument('--optimize', action='store_true', help='Run optuna hyperparam optimization')
    args = parser.parse_args()

    models_dir = 'ultimate_models'
    rewards_dir = 'ultimate_rewards'
    maybe_make_dir(models_dir)
    maybe_make_dir(rewards_dir)

    # If HPC for hyperparams
    if args.optimize:
        study = optuna.create_study(direction='minimize')
        study.optimize(hyperparam_optimization, n_trials=20)
        print("Best trial:", study.best_trial)
        return

    # LOAD DATA
    if args.resolution=='daily':
        price_df = load_price_data()
    else:
        price_df = load_intraday_data(args.resolution)
    fund_df = load_fundamental_data()
    sent_df = load_sentiment_data()
    combo_df = merge_data(price_df, fund_df, sent_df)
    n_total = len(combo_df)

    # create train/test or walk-forward
    if args.mode=='walkforward':
        window_size = int(n_total*0.5)
        test_size = int(n_total*0.1)
        splits = []
        start = 0
        while start+window_size+test_size <= n_total:
            tr = combo_df.iloc[start:start+window_size]
            te = combo_df.iloc[start+window_size:start+window_size+test_size]
            splits.append((tr, te))
            start+= test_size
    else:
        half = n_total//2
        train_df = combo_df.iloc[:half]
        test_df = combo_df.iloc[half:]

    env_params = {
        'initial_investment': 20000,
        'transaction_cost_pct': 0.001,
        'slippage_pct': 0.001,
        'stop_loss_pct': 0.1,
        'risk_per_trade': 0.1,
        'short_borrow_fee': 0.0002,
        'churn_penalty': 0.0,
        'resolution': args.resolution,
        'live_mode': args.live,
    }

    agent_params = {
        'gamma': args.gamma,
        'epsilon_decay': args.epsilon_decay,
        'lr': args.lr,
        'batch_size': 64
    }

    if args.mode=='walkforward':
        results = walkforward_eval(agent_params, env_params, splits, episodes=50)
        print("Walk-forward results:", results)
        np.save(os.path.join(rewards_dir,'walkforward_values.npy'), results)
        return

    # Single train/test
    if args.mode=='train':
        env = MultiAssetEnv(train_df, **env_params)
    else:
        env = MultiAssetEnv(test_df, **env_params)

    s_size = env.state_dim
    a_size = len(env.action_space)

    # Instantiate agent
    if args.agent=='continuous':
        def env_fn():
            return MultiAssetEnv(train_df, **env_params)
        agent = PPOAgent(env_fn, lr=args.lr, gamma=args.gamma, verbose=0)
    elif args.agent=='ensemble':
        # Two sub-agents
        ag1 = DoubleDQNAgent(s_size, a_size, gamma=args.gamma, epsilon_decay=args.epsilon_decay, lr=args.lr)
        ag2 = DoubleDQNAgent(s_size, a_size, gamma=args.gamma, epsilon_decay=args.epsilon_decay, lr=args.lr)
        agent = EnsembleAgent([ag1, ag2])
    else:
        agent = DoubleDQNAgent(s_size, a_size, gamma=args.gamma, epsilon_decay=args.epsilon_decay, lr=args.lr)

    scaler = get_scaler(env, num_samples=2000)

    # If test => load
    if args.mode=='test' and args.agent!='continuous':
        with open(os.path.join(models_dir,'scaler.pkl'),'rb') as f:
            scaler = pickle.load(f)
        agent.load(os.path.join(models_dir,'ultimate_dqn'))
        if hasattr(agent, 'epsilon'):
            agent.epsilon=0.01

    tb_writer = setup_tensorboard()

    portfolio_values = []
    for ep in range(args.num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, scaler, mode=args.mode, tb_writer=tb_writer, ep=ep)
        dt = datetime.now()-t0
        print(f"[{args.mode.upper()}] Episode {ep+1}/{args.num_episodes}, End Value={val:.2f}, Duration={dt}")
        log_metrics(tb_writer, {'portfolio_value': val}, ep+1)
        portfolio_values.append(val)

    if args.mode=='train' and args.agent!='continuous':
        agent.save(os.path.join(models_dir,'ultimate_dqn'))
        with open(os.path.join(models_dir,'scaler.pkl'),'wb') as f:
            pickle.dump(scaler, f)

    np.save(os.path.join(rewards_dir,f"{args.mode}_portfolio_values.npy"), portfolio_values)
    print("Done.")


if __name__=='__main__':
    main()