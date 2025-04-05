# Play module
import numpy as np
import tensorflow as tf

def play_one_episode(agent, env, scaler, mode='train', tb_writer=None, ep=0):
    """
    If tb_writer is not None, logs Q-value distribution for discrete agents each episode.
    """
    state = env.reset()
    state_scaled = scaler.transform([state])
    done = False
    total_reward = 0.0
    q_vals_list = []

    while not done:
        action = agent.act(state_scaled)

        # If discrete agent => log Q-values if needed
        if hasattr(agent, 'model'):
            q_ = agent.model.predict(state_scaled, verbose=0)
            q_vals_list.append(q_[0])

        next_state, reward, done, info = env.step(action)
        next_state_scaled = scaler.transform([next_state])
        total_reward+= reward

        if mode=='train' and hasattr(agent, 'store_experience'):
            agent.store_experience(state_scaled[0], action, reward, next_state_scaled[0], float(done))

        state_scaled = next_state_scaled

    if tb_writer and hasattr(agent, 'model') and len(q_vals_list)>0:
        with tb_writer.as_default():
            avg_q = np.mean(q_vals_list)
            tf.summary.scalar('avg_q_value', avg_q, step=ep)
            tf.summary.histogram('q_values', np.concatenate(q_vals_list), step=ep)
            tf.summary.scalar('episode_reward', total_reward, step=ep)
            tb_writer.flush()

    return info['cur_val']

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