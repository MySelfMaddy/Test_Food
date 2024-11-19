import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { Image } from 'image-js';

// Helper function to load images and resize them
async function loadImage(imagePath: string) {
    const img = await Image.load(imagePath);
    const resized = img.resize({ width: 224, height: 224 });
    const tensor = tf.browser.fromPixels(resized.getCanvas()).toFloat();
    return tensor.expandDims(0); // Add batch dimension
}

// Load images and labels for the dataset
async function loadData(imagePaths: string[], label: number) {
    const imageTensors: tf.Tensor[] = [];
    const labels: number[] = [];

    for (const imagePath of imagePaths) {
        const tensor = await loadImage(imagePath);
        imageTensors.push(tensor);
        labels.push(label);
    }

    return { images: tf.concat(imageTensors), labels: tf.tensor(labels) };
}

// Define a simple CNN model in TensorFlow.js
function createModel() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [224, 224, 3],
        kernelSize: 3,
        filters: 32,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // Binary classification

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

// Main function to train the model
async function trainModel() {
    const trainFreshImages = fs.readdirSync(path.join('train', 'fresh')).map((file) => path.join('train', 'fresh', file));
    const trainSpoiledImages = fs.readdirSync(path.join('train', 'spoiled')).map((file) => path.join('train', 'spoiled', file));

    const validationFreshImages = fs.readdirSync(path.join('validation', 'fresh')).map((file) => path.join('validation', 'fresh', file));
    const validationSpoiledImages = fs.readdirSync(path.join('validation', 'spoiled')).map((file) => path.join('validation', 'spoiled', file));

    const { images: trainFresh, labels: labelFresh } = await loadData(trainFreshImages, 0);
    const { images: trainSpoiled, labels: labelSpoiled } = await loadData(trainSpoiledImages, 1);

    const { images: validationFresh, labels: validationLabelFresh } = await loadData(validationFreshImages, 0);
    const { images: validationSpoiled, labels: validationLabelSpoiled } = await loadData(validationSpoiledImages, 1);

    const trainImages = tf.concat([trainFresh, trainSpoiled]);
    const trainLabels = tf.concat([labelFresh, labelSpoiled]);

    const validationImages = tf.concat([validationFresh, validationSpoiled]);
    const validationLabels = tf.concat([validationLabelFresh, validationLabelSpoiled]);

    const model = createModel();

    // Train the model
    await model.fit(trainImages, trainLabels, {
        epochs: 10,
        batchSize: 32,
        validationData: [validationImages, validationLabels],
        callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 })
    });

    // Save the model in TensorFlow.js format
    await model.save('file://./food_freshness_model');
}

trainModel();
