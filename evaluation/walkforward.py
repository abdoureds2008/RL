# Walkforward module
def walkforward_eval(agent_params, env_params, data_splits, episodes=50):
    results = []
    for i,(train_df,test_df) in enumerate(data_splits):
        print(f"Walk-forward {i+1}/{len(data_splits)}")
        env_train = MultiAssetEnv(train_df, **env_params)
        env_test = MultiAssetEnv(test_df, **env_params)
        state_size = env_train.state_dim
        action_size = len(env_train.action_space)

        agent = DoubleDQNAgent(
            state_size, action_size,
            gamma=agent_params['gamma'],
            epsilon_decay=agent_params['epsilon_decay'],
            lr=agent_params['lr'],
            batch_size=agent_params['batch_size']
        )

        sc = get_scaler(env_train, num_samples=2000)
        # train
        for e in range(episodes):
            play_one_episode(agent, env_train, sc, mode='train')
        # test
        val = play_one_episode(agent, env_test, sc, mode='test')
        results.append(val)
    return results