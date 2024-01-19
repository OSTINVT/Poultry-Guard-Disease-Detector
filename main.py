import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Load CSV file containing image filenames and labels
csv_path = "C:\\Users\\ostin v thomas\\Desktop\\PROJECTS\\PoultryGuard-Disease_Detector\\dis_data.csv"
df = pd.read_csv(csv_path)

# Define image and label columns
image_column = 'images'
label_column = 'label'

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set up data generators
batch_size = 32
train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    directory="C:\\Users\\ostin v thomas\\Desktop\\PROJECTS\\PoultryGuard-Disease_Detector\\image_data",
                                                    x_col=image_column,
                                                    y_col=label_column,
                                                    target_size=(224, 224),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                  directory="C:\\Users\\ostin v thomas\\Desktop\\PROJECTS\\PoultryGuard-Disease_Detector\\image_data",
                                                  x_col=image_column,
                                                  y_col=label_column,
                                                  target_size=(224, 224),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(df[label_column])
num_classes = len(label_encoder.classes_)

# Build the model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
model.fit(train_generator, epochs=epochs, validation_data=test_generator)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy}')
