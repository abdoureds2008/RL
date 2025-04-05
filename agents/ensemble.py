# Ensemble module
import numpy as np

class EnsembleAgent:
    """
    Example: Weighted or majority vote across multiple discrete agents.
    """
    def __init__(self, agents):
        self.agents = agents

    def act(self, state):
        # Weighted average or majority vote
        # For simplicity, do average Q-values
        q_vals = [ag.model.predict(state, verbose=0) for ag in self.agents]
        avg_q = np.mean(q_vals, axis=0)
        return np.argmax(avg_q[0])

    def store_experience(self, s, a, r, s2, d):
        for ag in self.agents:
            ag.store_experience(s, a, r, s2, d)

    def replay(self):
        for ag in self.agents:
            ag.replay()

    def save(self, path):
        for i,ag in enumerate(self.agents):
            ag.save(path+f"_agent{i}")

    def load(self, path):
        for i,ag in enumerate(self.agents):
            ag.load(path+f"_agent{i}")
