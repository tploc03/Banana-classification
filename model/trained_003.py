import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks #type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

img_size = (150, 150)
batch_size = 32
learning_rate = 0.0001
epochs = 50

train_dir = "raw/"
test_dir = "raw_to_test/"

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="training"
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="validation"
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True
)

# Display images from the dataset
def display_images(dataset, num_images=9, dataset_name="Dataset"):
    class_names = dataset.class_names 
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"Images from {dataset_name}", fontsize=16)

    for images, labels in dataset.take(1): 
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[int(labels[i])])
            plt.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


display_images(train_dataset, num_images=9, dataset_name="Train")
display_images(val_dataset, num_images=9, dataset_name="Validation")
display_images(test_dataset, num_images=9, dataset_name="Test")


class_names = train_dataset.class_names
input_shape = img_size + (3,) #150x150x3
class_count = len(class_names)

model = models.Sequential(name="banana_classifier_trained_003")

model.add(layers.Rescaling(1./255, input_shape=input_shape))

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))  # Giảm 25% nơron ngẫu nhiên

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))  # Thêm Dropout

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout mạnh hơn ở tầng ẩn cuối

model.add(layers.Dense(class_count, activation='softmax'))



model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping if validation loss does not improve
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stop]
)

model.summary()

# Save model
os.makedirs('./output', exist_ok=True)
model.save('./output/banana_classifier_trained_003.keras')

# Evaluate model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f} - Test Loss: {test_loss:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Confusion matrix
true_labels, predicted_labels = [], []
for images, labels in test_dataset:
    preds = model.predict(images)
    true_labels.extend(labels.numpy())
    predicted_labels.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(12, 12))
for images, labels in test_dataset.take(1):  #1 batch from test_dataset
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    for i in range(9):  # show 9 first images 
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted_classes[i]]
        color = "green" if true_label == pred_label else "red"
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
        plt.axis("off")

plt.tight_layout()
plt.show()
