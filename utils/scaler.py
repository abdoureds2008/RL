# Scaler module
def get_scaler(env, num_samples=2000):
    """
    Collect random states by stepping env with random actions.
    Then fit a StandardScaler.
    """
    states = []
    s = env.reset()
    for _ in range(num_samples):
        a = np.random.choice(env.action_space)
        s, _, done, _ = env.step(a)
        states.append(s)
        if done:
            s = env.reset()
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)