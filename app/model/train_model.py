from keras import layers, models
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import yaml
import os

def create_cnn_model(input_shape=(224, 224, 3), num_classes=3):
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(data_dir, epochs=20, batch_size=32):
    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation and testing
    valid_test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'Training'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )


    # Load validation data
    validation_generator = valid_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'Validation'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load test data
    test_generator = valid_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'Testing'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Create and compile model
    model = create_cnn_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Model checkpoint callback
    checkpoint_path = "model/potato_disease_model.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint_callback, early_stopping]
    )

    return model, history

def plot_training_history(history):
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['Train', 'Validation'])
    
    # Loss plot
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def evaluate_model(model, test_generator):
    """Evaluate the model on test data and print metrics"""
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest Dataset Metrics:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Get predictions for confusion matrix
    predictions = model.predict(test_generator)
    predicted_classes = predictions.argmax(axis=1)
    true_classes = test_generator.classes

    # Print classification report
    class_names = list(test_generator.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Directory containing the dataset
    # Dataset should be organized as:
    # data/
    #   ├── Training/
    #   │   ├── Early_Blight/
    #   │   ├── Late_Blight/
    #   │   └── Healthy/
    #   ├── Validation/
    #   │   ├── Early_Blight/
    #   │   ├── Late_Blight/
    #   │   └── Healthy/
    #   └── Testing/
    #       ├── Early_Blight/
    #       ├── Late_Blight/
    #       └── Healthy/
    DATA_DIR = "data"
    
    # Train the model
    model, history = train_model(DATA_DIR)
    
    # Plot training history
    plot_training_history(history)
    
    # Print model summary
    model.summary()
    
    # Evaluate on test set
    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_DIR, 'Testing'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    evaluate_model(model, test_generator)
