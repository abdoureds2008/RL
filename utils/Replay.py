# 5) EXPERIENCE REPLAY
############################
class ReplayBuffer:
    def __init__(self, obs_dim, size=10000):
        self.obs1_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.int32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.int32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, s, a, r, s2, d):
        idx = self.ptr
        self.obs1_buf[idx] = s
        self.obs2_buf[idx] = s2
        self.acts_buf[idx] = a
        self.rews_buf[idx] = r
        self.done_buf[idx] = d

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            s=self.obs1_buf[idxs],
            a=self.acts_buf[idxs],
            r=self.rews_buf[idxs],
            s2=self.obs2_buf[idxs],
            d=self.done_buf[idxs],
        )


