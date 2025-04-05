# Lstm_model module

def create_lstm_model(state_dim, action_dim, hidden=64, dropout=0.2, lr=1e-3):
    inp = Input(shape=(state_dim,))
    x = Reshape((1, state_dim))(inp)
    x = LSTM(hidden, activation='tanh')(x)
    x = Dropout(dropout)(x)
    out = Dense(action_dim, activation='linear')(x)
    model = Model(inp, out)
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr, clipnorm=1.0))
    model.summary()
    return model
