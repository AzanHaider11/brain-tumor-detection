import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt



class BrainTumorDetector:
    def __init__(self, img_height=150, img_width=150):

        self.img_height = img_height
        self.img_width = img_width
        self.model = None

    def prepare_data(self, data_dir):

        # Data augmentation and preprocessing
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # 20% for validation
        )

        # Training data generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )

        # Validation data generator
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )

        return train_generator, validation_generator

    def build_model(self):

        self.model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', 
                   input_shape=(self.img_height, self.img_width, 3)),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Flatten and Fully Connected Layers
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification
        ])

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def train_model(self, train_generator, validation_generator, epochs=50):

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )

        model_checkpoint = ModelCheckpoint(
            'best_brain_tumor_model.keras', 
            monitor='val_accuracy', 
            save_best_only=True
        )

        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping, model_checkpoint]
        )

        return history

    def evaluate_model(self, validation_generator):

        return self.model.evaluate(validation_generator)

    def plot_training_history(self, history):

        plt.figure(figsize=(12, 4))
        
        # Accuracy subplot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Loss subplot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.show()

def main():
    
    DATASET_PATH = 'data'

    # Initialize Brain Tumor Detector
    detector = BrainTumorDetector()
    
    # Prepare data
    print("Preparing data...")
    train_generator, validation_generator = detector.prepare_data(DATASET_PATH)
    
    # Build model
    print("Building model...")
    model = detector.build_model()
    
    # Train model
    print("Training model...")
    history = detector.train_model(train_generator, validation_generator)
    
    # Plot training history
    detector.plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    evaluation = detector.evaluate_model(validation_generator)
    print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

if __name__ == "__main__":
    main()