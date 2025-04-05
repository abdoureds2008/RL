# Ppo_agent module
class PPOAgent:
    def __init__(self, env_fn, policy='MlpPolicy', lr=1e-3, gamma=0.99, verbose=1):
        self.env = DummyVecEnv([env_fn])
        self.model = PPO(policy, self.env, learning_rate=lr, gamma=gamma, verbose=verbose)

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)

    def act(self, obs):
        action, _ = self.model.predict(obs)
        return action[0]

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, env=self.env)
