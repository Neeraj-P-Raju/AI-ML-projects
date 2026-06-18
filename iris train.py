from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# One-hot encoding
y = to_categorical(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tiny network
model = Sequential([
    Dense(4, activation='sigmoid', input_shape=(4,)),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=200,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test)

print("Accuracy =", acc)

# Extract weights
weights = model.get_weights()

for i, w in enumerate(weights):
    print("\nLayer", i)
    print(w)