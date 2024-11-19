import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { fromEvent } from 'rxjs';
import { map } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class ModelServiceService {

  constructor() { }

  // Helper function to load images and resize them
  async loadImage(imageFile: File) {
    const img = await this.readImageAsDataURL(imageFile);
    const imgElement = new Image();
    imgElement.src = img;
    await new Promise(resolve => imgElement.onload = resolve);

    const tensor = tf.browser.fromPixels(imgElement).resizeBilinear([224, 224]).toFloat();
    return tensor.expandDims(0); // Add batch dimension
  }

  // Function to read image as a data URL
  readImageAsDataURL(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  // Load images and labels for the dataset
  async loadData(imageFiles: File[], label: number) {
    const imageTensors: tf.Tensor[] = [];
    const labels: number[] = [];

    for (const imageFile of imageFiles) {
      const tensor = await this.loadImage(imageFile);
      imageTensors.push(tensor);
      labels.push(label);
    }

    return { images: tf.concat(imageTensors), labels: tf.tensor(labels) };
  }

  // Define a simple CNN model in TensorFlow.js
  createModel() {
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
  async trainModel(freshFiles: File[], spoiledFiles: File[]) {
    const { images: trainFresh, labels: labelFresh } = await this.loadData(freshFiles, 0);
    const { images: trainSpoiled, labels: labelSpoiled } = await this.loadData(spoiledFiles, 1);

    const trainImages = tf.concat([trainFresh, trainSpoiled]);
    const trainLabels = tf.concat([labelFresh, labelSpoiled]);

    // Create the model
    const model = this.createModel();

    // Train the model
    await model.fit(trainImages, trainLabels, {
      epochs: 10,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 })
    });

    // Save the model (in-browser or to a server endpoint)
    await model.save('indexeddb://food_freshness_model');
    await model.save('file://assets');
  }

  // Function to handle file input change
  handleFileInputChange(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input && input.files) {
      const freshFiles: File[] = [];
      const spoiledFiles: File[] = [];

      // Assume files are passed from the input, separate them by class (fresh/spoiled)
      Array.from(input.files).forEach((file: File) => {
        // Example condition: file name indicates class (adjust as needed)
        if (file.name.includes('fresh')) {
          freshFiles.push(file);
        } else if (file.name.includes('spoiled')) {
          spoiledFiles.push(file);
        }
      });

      // Train model using these files
      this.trainModel(freshFiles, spoiledFiles);
    }
  }
}
