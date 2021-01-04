import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#<------Importation des données------->
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

taille_image = [28, 28, 1] #28x28 pixels sur 1 cannal (nuance de gris)

#<------Fonction d'afichage des prédictions merci à Aurélien Géron------->
def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

#<------Création du modèle------->
generateur = keras.models.Sequential([
    #7x7x128
    keras.layers.Dense(7*7*128, input_shape=[100]),
    keras.layers.Reshape([7, 7, 128]),
    keras.layers.BatchNormalization(),
    #14x14x64
    keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", activation="selu"),
    keras.layers.BatchNormalization(),
    #28x28x1
    keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", activation="tanh")
]
)

discriminateur = keras.models.Sequential([
    #14x14x64
    keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same", activation=keras.layers.LeakyReLU(0.2),
                        input_shape=[28, 28, 1]),
    keras.layers.Dropout(0.4),
    #28x28x128
    keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same", activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    #28x28x1
    keras.layers.Dense(1, activation="sigmoid")
])

dcgan = keras.models.Sequential([generateur, discriminateur])
discriminateur.compile(loss="binary_crossentropy", optimizer="rmsprop")
#On gèle les poids du discriminateur
discriminateur.trainable = False
dcgan.compile(loss="binary_crossentropy", optimizer="rmsprop")
#traitement des données pour avoir la bonne forme
X_train_dcgan = X_train.reshape(-1, 28, 28, 1)* 2. -1

#<------Création de la boucle d'entrainement------->
#On va utiliser une pipeline d'entrainement pour des problèmes éventuels de dépassement en RAM
taille_lots = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train_dcgan).shuffle(500)
dataset = dataset.batch(taille_lots, drop_remainder=True).prefetch(1)

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))
        for X_batch in dataset:
            # phase 1 - Entrainement du discriminateur
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - Emtrainement du générateur
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)

    plot_multiple_images(generated_images, 8)
    plt.show()
#<------Entrainement------->

train_gan(dcgan, dataset, taille_lots, 100)

dcgan.save('dcgan')