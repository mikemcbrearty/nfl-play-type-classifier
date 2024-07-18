import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_data
from tensorflow import keras
from tensorflow.keras import layers

# Neural network model using Keras

(train_data, val_data, test_data), (train_targets, val_targets, test_targets) = load_data()

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
callbacks = [
    keras.callbacks.ModelCheckpoint("play_data_dense.keras", save_best_only=True),
]
history = model.fit(
    train_data,
    train_targets,
    epochs=80,
    batch_size=512,
    validation_data=(val_data, val_targets),
    callbacks=callbacks,
    verbose=0,
)

def plot_training_loss_acc(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.clf()
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

results = model.evaluate(test_data, test_targets)
print(results)

plot_training_loss_acc()
