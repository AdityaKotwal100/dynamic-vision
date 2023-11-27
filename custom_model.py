import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories for train, validation, and test datasets
train_dir = "path_to_train_directory"
validation_dir = "path_to_validation_directory"
test_dir = "path_to_test_directory"

# Define parameters
num_classes = 10  # Replace with the number of classes in your dataset
batch_size = 32
num_epochs = 10

# Load pre-trained VGG16 model without classification layers
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the VGG16 base
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(
    Dense(num_classes, activation="softmax")
)  # num_classes is the number of output classes

# Compile the model
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# ImageDataGenerator for data augmentation and loading images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Flow training images in batches using ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=batch_size, class_mode="categorical"
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
)

# Calculate steps per epoch and validation steps
train_steps = train_generator.n // train_generator.batch_size
val_steps = validation_generator.n // validation_generator.batch_size

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=val_steps,
)

# Evaluate the model using the test dataset
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=batch_size, class_mode="categorical"
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")
