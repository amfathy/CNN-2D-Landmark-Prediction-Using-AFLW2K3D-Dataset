import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Load AFLW2K dataset
import tensorflow_datasets as tfds

# Load dataset
ds = tfds.load('aflw2k3d', split='train')

# Collect images and landmarks
images = []
landmarks2D = []

#the dataset
for ex in ds.take(2000):
    images.append(ex['image'])
    # Flatten the 68x2 array of landmarks into a 1D array
    landmarks = np.reshape(ex['landmarks_68_3d_xy_normalized'], (-1,))
    landmarks2D.append(landmarks)


images = np.array(images)
landmarks2D = np.array(landmarks2D)


train_images = images[:1500]
train_landmarks = landmarks2D[:1500]
test_images = images[1500:]
test_landmarks = landmarks2D[1500:]

# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(450, 450, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(136)  # Output layer with 68 (x, y) coordinates
])

# Compile
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(train_images, train_landmarks, epochs=3, batch_size=64, validation_split=0.1)

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Evaluate the model on test data
test_loss = model.evaluate(test_images, test_landmarks)
print("Test Loss:", test_loss)
print("Number of Landmarks:", landmarks2D.shape[1])
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Load AFLW2K dataset
import tensorflow_datasets as tfds

# Load dataset
ds = tfds.load('aflw2k3d', split='train')

# Collect images and landmarks
images = []
landmarks2D = []

#the dataset
for ex in ds.take(2000):
    images.append(ex['image'])
    # Flatten the 68x2 array of landmarks into a 1D array
    landmarks = np.reshape(ex['landmarks_68_3d_xy_normalized'], (-1,))
    landmarks2D.append(landmarks)


images = np.array(images)
landmarks2D = np.array(landmarks2D)


train_images = images[:1500]
train_landmarks = landmarks2D[:1500]
test_images = images[1500:]
test_landmarks = landmarks2D[1500:]

# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(450, 450, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(136)  # Output layer with 68 (x, y) coordinates
])

# Compile
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(train_images, train_landmarks, epochs=3, batch_size=64, validation_split=0.1)

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Evaluate the model on test data
test_loss = model.evaluate(test_images, test_landmarks)
print("Test Loss:", test_loss)
print("Number of Landmarks:", landmarks2D.shape[1])
