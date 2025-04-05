# Double_dqn module

class DoubleDQNAgent:
    """
    Double DQN with LSTM for discrete actions.
    Trains every 'train_freq' environment steps.
    """
    def __init__(self,
                 state_size,
                 action_size,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 buffer_size=10000,
                 batch_size=32,
                 tau=0.1,
                 lr=1e-3,
                 train_freq=1):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.train_freq = train_freq

        self.memory = ReplayBuffer(obs_dim=state_size, size=buffer_size)
        self.model = create_lstm_model(state_size, action_size, hidden=64, dropout=0.2, lr=lr)
        self.target_model = create_lstm_model(state_size, action_size, hidden=64, dropout=0.2, lr=lr)
        self.update_target_network(tau=1.0)
        self.step_count = 0

    def update_target_network(self, tau=None):
        if tau is None:
            tau = self.tau
        main_w = self.model.get_weights()
        targ_w = self.target_model.get_weights()
        new_w = []
        for mw, tw in zip(main_w, targ_w):
            new_w.append(tau*mw + (1-tau)*tw)
        self.target_model.set_weights(new_w)

    def store_experience(self, s, a, r, s2, d):
        self.memory.store(s, a, r, s2, d)
        self.step_count+=1
        if self.step_count%self.train_freq==0:
            self.replay()

    def act(self, state):
        if np.random.rand()<self.epsilon:
            return np.random.randint(self.action_size)
        q = self.model.predict(state, verbose=0)
        return np.argmax(q[0])

    def replay(self):
        if self.memory.size<self.batch_size:
            return
        minibatch = self.memory.sample_batch(self.batch_size)
        s = minibatch['s']
        a = minibatch['a']
        r = minibatch['r']
        s2 = minibatch['s2']
        d = minibatch['d']

        # DoubleDQN
        q_next_online = self.model.predict(s2, verbose=0)
        acts_next = np.argmax(q_next_online, axis=1)
        q_next_target = self.target_model.predict(s2, verbose=0)
        target = r + (1-d)*self.gamma * q_next_target[np.arange(self.batch_size), acts_next]

        q_curr = self.model.predict(s, verbose=0)
        q_curr[np.arange(self.batch_size), a] = target

        self.model.train_on_batch(s, q_curr)

        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay

        self.update_target_network()

    def save(self, name):
        self.model.save_weights(name + "_main.weights.h5")
        self.target_model.save_weights(name + "_target.weights.h5")

    def load(self, name):
        self.model.load_weights(name + "_main.weights.h5")
        self.target_model.load_weights(name + "_target.weights.h5")
