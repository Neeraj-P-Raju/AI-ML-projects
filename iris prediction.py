import math

# Layer 0 (Input -> Hidden)
W1 = [
 [-0.39588213, 0.1537606,  0.6252354,  -0.348488  ],
 [ 0.33393356, 0.42985642, 0.69137114, -0.60241693],
 [ 0.28938168, 0.7759167,  0.74146724,  1.0135716 ],
 [ 0.6402441,  0.16270967, 0.43608624,  0.9027375 ]
]

B1 = [
 -0.30078977,
  0.4301676,
  0.32961786,
  0.02188457
]

# Layer 2 (Hidden -> Output)
W2 = [
 [-0.16593522, -0.69135934,  1.0806979 ],
 [ 0.88939184, -0.17831251,  0.03712272],
 [ 0.24036434,  0.6074807,  -0.74844354],
 [-1.7610949,   0.52938336,  0.3780825 ]
]

B2 = [
 0.08714361,
 -0.12751965,
 0.0294865
]

# -----------------------------

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def softmax(z):

    exps = []

    for v in z:
        exps.append(math.exp(v))

    s = sum(exps)

    probs = []

    for e in exps:
        probs.append(e / s)

    return probs

# -----------------------------

def predict(x):

    # Hidden layer
    hidden = []

    for j in range(4):

        h = B1[j]

        for i in range(4):
            h += x[i] * W1[i][j]

        hidden.append(sigmoid(h))

    # Output layer
    output = []

    for k in range(3):

        o = B2[k]

        for j in range(4):
            o += hidden[j] * W2[j][k]

        output.append(o)

    probs = softmax(output)

    return probs

# -----------------------------
# TEST SAMPLE
# -----------------------------

sample = [5.1, 3.5, 1.4, 0.2]

result = predict(sample)

print("Probabilities:", result)

flower = result.index(max(result))

if flower == 0:
    print("Setosa")

elif flower == 1:
    print("Versicolor")

else:
    print("Virginica")