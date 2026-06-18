import math

# Activation functions
def relu(x):
    return x if x > 0 else 0

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# Dense layer
def dense(inputs, weights, bias):
    outputs = []

    for j in range(len(bias)):
        s = bias[j]
        for i in range(len(inputs)):
            s += inputs[i] * weights[i][j]
        outputs.append(s)

    return outputs

# Example network weights
W1 = [[0.1 for _ in range(10)] for _ in range(10)]
B1 = [0.0] * 10

W2 = [[0.1 for _ in range(5)] for _ in range(10)]
B2 = [0.0] * 5

W3 = [
    [0.5],
    [-0.3],
    [0.8],
    [0.2],
    [-0.4]
]
B3 = [0.0]

def predict(x):
    # Hidden layer 1
    h1 = dense(x, W1, B1)
    h1 = [relu(v) for v in h1]

    # Hidden layer 2
    h2 = dense(h1, W2, B2)
    h2 = [relu(v) for v in h2]

    # Output layer
    out = dense(h2, W3, B3)[0]
    out = sigmoid(out)

    return out

# Example input (10 features)
sample = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0
]

prediction = predict(sample)

print("Prediction:", prediction)

if prediction > 0.5:
    print("will rain")
else:
    print("no rain")