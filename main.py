import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

if __name__ == '__main__':

    data_dir = pathlib.Path("D://AAinfa/sem2/AnObCyfr/trainset")
    print("\ndata_dir :\n")
    print(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("\nimage_count :\n")
    print(image_count)

    batch_size = 32
    img_height = 100
    img_width = 100
# podzielenie obrazkow na zbiór treningowy oraz zbiór validacji 80 do 20
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print("\nDataset :\n")
    print(class_names)

    import matplotlib.pyplot as plt

# Pierwsze 9 obrazkow z bazy treningowej
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    # W razie potrzeby mozna iterowac recznie
    for image_batch, labels_batch in train_ds:
        print("\nThis is a batch of 32 images of shape 100x100x3 (the last dimension refers to color channels RGB) :\n")
        print(image_batch.shape)
        print(labels_batch.shape)
        break
    # Image Batch jest to tensor o kształcie (32, 100, 100, 3). Na ten batch składa się 32 obrazki

    # Automatyczna konfiguracja dataset w celach wydajnosciowych
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


    # Standaryzowanie danych
    # Wartości RGB w zakresie [0,255] nie są najlepsze dla sieci nauronowych,
    # dlatego standaryzujemy wartosci do zakresu [0, 1]

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print("\n pixele sa od 0.0 do 1.0 \n")
    print(np.min(first_image), np.max(first_image))


    # Tworzenie modelu

    num_classes = len(class_names)
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)

        "test"
    ])


    # Komplitacja modelu
    # Optymalizacja Adama to stochastyczna metoda opadania gradientu
    # oparta na adaptacyjnej estymacji momentów pierwszego i drugiego rzędu.

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    # Trenowanie modelu
    epochs = 5
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Wizualizacja wynikow treningu - dwa wykresy

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # testowanie nowymi danymi
    corn_path = ("D://AAinfa/sem2/AnObCyfr/trainset/testowyCorn.jpg")

    img = tf.keras.utils.load_img(
        corn_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )