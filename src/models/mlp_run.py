from mlp.mlp import MLP
from mlp.layers import Dense

model = MLP([
    Dense(neurons=10, input_size=4, activation='relu'),
    Dense(neurons=8, activation='relu'),
    Dense(neurons=3, activation='softmax')
])

model.summary()