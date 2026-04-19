import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing import image


class Preprocess:
    def __init__(self, image_size=(224, 224), batch_size=32, augment=True):
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment

    def load_data(self, data_dir):
        if self.augment:
            datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
            )
        else:
            datagen = ImageDataGenerator(rescale=1./255)

        generator = datagen.flow_from_directory(
            data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        return generator


class CNN_Model:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        return self.model

    def train_model(self, train_generator, validation_generator, epochs=10):
        history = self.model.fit(train_generator,
                                 steps_per_epoch=len(train_generator),
                                 validation_data=validation_generator,
                                 validation_steps=len(validation_generator),
                                 epochs=epochs)
        return history

    def evaluate_model(self, test_generator):
        test_loss, test_accuracy = self.model.evaluate(test_generator)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        return test_loss, test_accuracy

    def predict(self, image_path, class_names):
        img = image.load_img(image_path, target_size=self.input_shape[:2])
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array / 255.0, axis=0)
        predictions = self.model.predict(img_array)
        return class_names[np.argmax(predictions)], np.max(predictions)

    def save_model(self, model_path):
        self.model.save(model_path)
        print(f"Model saved to {model_path}")


class ImageNet_Model:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model = MobileNetV2(
            weights='imagenet', include_top=False, input_shape=self.input_shape)
        self.base_model.trainable = False
        self.model = self.build_model()

    def build_model(self):
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=self.base_model.input, outputs=predictions)
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        return self.model

    def train_model(self, train_generator, validation_generator, epochs=10):
        history = self.model.fit(train_generator,
                                 steps_per_epoch=len(train_generator),
                                 validation_data=validation_generator,
                                 validation_steps=len(validation_generator),
                                 epochs=epochs)
        return history

    def fine_tune(self, layers_to_unfreeze=20):
        for layer in self.base_model.layers[-layers_to_unfreeze:]:
            layer.trainable = True
        print(f"Unfroze last {layers_to_unfreeze} layers for fine-tuning.")

    def predict(self, image_path, class_names):
        img = image.load_img(image_path, target_size=self.input_shape[:2])
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array / 255.0, axis=0)
        predictions = self.model.predict(img_array)
        return class_names[np.argmax(predictions)], np.max(predictions)

    def evaluate_model(self, test_generator):
        test_loss, test_accuracy = self.model.evaluate(test_generator)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        return test_loss, test_accuracy

    def save_model(self, model_path):
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
