import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';

@Injectable({
  providedIn: 'root'
})
export class FoodQualityService {
  private model: tf.LayersModel | null = null;
  private readonly MODEL_KEY = 'local-food-quality-model';

  constructor() {
    this.loadModel();
  }

  // Load the model, checking IndexedDB first, then TensorFlow Hub if not found
  async loadModel() {
    try {
      // Try loading model from IndexedDB
      this.model = await tf.loadLayersModel(`indexeddb://${this.MODEL_KEY}`);
      console.log('Model loaded from IndexedDB');
    } catch (error) {
      console.log('Model not found in IndexedDB, loading from TensorFlow Hub');

      // Load the model from TensorFlow Hub and save it locally
      const mobilenet = await tf.loadLayersModel(
        'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1',
        { fromTFHub: true }
      );

      // Rebuild model with custom output layer
      const layer = mobilenet.getLayer('global_average_pooling2d');
      this.model = tf.sequential({
        layers: [
          tf.layers.inputLayer({ inputShape: [224, 224, 3] }),
          layer,
          tf.layers.dense({ units: 1, activation: 'sigmoid' })
        ]
      });

      // Save the customized model to IndexedDB
      await this.model.save(`indexeddb://${this.MODEL_KEY}`);
      console.log('Model saved to IndexedDB');
    }

    // Compile the model (if it's newly loaded)
    await this.model?.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });
  }

  // Capture image, preprocess, and predict freshness
  async captureAndPredict(): Promise<string> {
    const image = await Camera.getPhoto({
      quality: 90,
      allowEditing: false,
      resultType: CameraResultType.DataUrl,
      source: CameraSource.Camera
    });

    const img = await this.processImage(image.dataUrl);
    const prediction = this.model?.predict(img) as tf.Tensor;
    const result = (await prediction.data())[0];

    return result > 0.5 ? 'Spoiled' : 'Fresh';
  }

  private async processImage(dataUrl: string): Promise<tf.Tensor4D> {
    const img = new Image();
    img.src = dataUrl;
    await new Promise((resolve) => {
      img.onload = resolve;
    });

    // Convert to tensor, resize, normalize, and expand dimensions
    let tensor = tf.browser.fromPixels(img).resizeBilinear([224, 224]);
    tensor = tensor.toFloat().div(tf.scalar(255)).expandDims();
    return tensor as tf.Tensor4D;
  }
}



================================================


import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';

@Injectable({
  providedIn: 'root'
})
export class FoodQualityService {
  private model: tf.LayersModel | null = null;

  constructor() {
    this.loadModel();
  }

  // Load MobileNet model with a custom output layer for binary classification
  async loadModel() {
    const mobilenet = await tf.loadLayersModel(
      'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1',
      { fromTFHub: true }
    );

    const layer = mobilenet.getLayer('global_average_pooling2d'); // Use last feature extraction layer
    const newModel = tf.sequential({
      layers: [
        tf.layers.inputLayer({ inputShape: [224, 224, 3] }),
        layer,
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
      ]
    });

    this.model = newModel;
    await this.model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });
  }

  // Capture and preprocess image from camera
  async captureAndPredict(): Promise<string> {
    const image = await Camera.getPhoto({
      quality: 90,
      allowEditing: false,
      resultType: CameraResultType.DataUrl,
      source: CameraSource.Camera
    });

    if (!this.model) {
      await this.loadModel();
    }

    const img = await this.processImage(image.dataUrl);
    const prediction = this.model?.predict(img) as tf.Tensor;
    const result = (await prediction.data())[0];

    return result > 0.5 ? 'Spoiled' : 'Fresh';
  }

  private async processImage(dataUrl: string): Promise<tf.Tensor4D> {
    const img = new Image();
    img.src = dataUrl;
    await new Promise((resolve) => {
      img.onload = resolve;
    });

    // Convert to tensor, resize, normalize, and expand dimensions
    let tensor = tf.browser.fromPixels(img).resizeBilinear([224, 224]);
    tensor = tensor.toFloat().div(tf.scalar(255)).expandDims();
    return tensor as tf.Tensor4D;
  }
}


==================================


import { Component } from '@angular/core';
import { FoodQualityService } from '../services/food-quality.service';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage {
  prediction: string | null = null;
  isProcessing = false;

  constructor(private foodQualityService: FoodQualityService) {}

  async captureAndClassify() {
    this.isProcessing = true;
    this.prediction = null;

    try {
      this.prediction = await this.foodQualityService.captureAndPredict();
    } catch (error) {
      console.error('Prediction failed', error);
    } finally {
      this.isProcessing = false;
    }
  }
}


==========================================

<ion-header>
  <ion-toolbar>
    <ion-title>Food Quality Checker</ion-title>
  </ion-toolbar>
</ion-header>

<ion-content class="ion-padding">
  <ion-button expand="full" (click)="captureAndClassify()" [disabled]="isProcessing">
    Check Food Freshness
  </ion-button>

  <div *ngIf="isProcessing">
    <ion-spinner></ion-spinner>
    <p>Processing...</p>
  </div>

  <div *ngIf="prediction && !isProcessing">
    <h2>Result: {{ prediction }}</h2>
  </div>
</ion-content>

===============================================



import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';

@Injectable({
  providedIn: 'root'
})
export class FoodQualityService {
  private model: tf.LayersModel | null = null;
  private readonly MODEL_KEY = 'local-food-quality-model';

  constructor() {
    this.loadModel();
  }

  // Load the model, checking IndexedDB first, then TensorFlow Hub if not found
  async loadModel() {
    try {
      // Try loading model from IndexedDB
      this.model = await tf.loadLayersModel(`indexeddb://${this.MODEL_KEY}`);
      console.log('Model loaded from IndexedDB');
    } catch (error) {
      console.log('Model not found in IndexedDB, loading from TensorFlow Hub');

      // Load the model from TensorFlow Hub and save it locally
      const mobilenet = await tf.loadLayersModel(
        'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1',
        { fromTFHub: true }
      );

      // Rebuild model with custom output layer
      const layer = mobilenet.getLayer('global_average_pooling2d');
      this.model = tf.sequential({
        layers: [
          tf.layers.inputLayer({ inputShape: [224, 224, 3] }),
          layer,
          tf.layers.dense({ units: 1, activation: 'sigmoid' })
        ]
      });

      // Save the customized model to IndexedDB
      await this.model.save(`indexeddb://${this.MODEL_KEY}`);
      console.log('Model saved to IndexedDB');
    }

    // Compile the model (if it's newly loaded)
    await this.model?.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });
  }

  // Capture image, preprocess, and predict freshness
  async captureAndPredict(): Promise<string> {
    const image = await Camera.getPhoto({
      quality: 90,
      allowEditing: false,
      resultType: CameraResultType.DataUrl,
      source: CameraSource.Camera
    });

    const img = await this.processImage(image.dataUrl);
    const prediction = this.model?.predict(img) as tf.Tensor;
    const result = (await prediction.data())[0];

    return result > 0.5 ? 'Spoiled' : 'Fresh';
  }

  private async processImage(dataUrl: string): Promise<tf.Tensor4D> {
    const img = new Image();
    img.src = dataUrl;
    await new Promise((resolve) => {
      img.onload = resolve;
    });

    // Convert to tensor, resize, normalize, and expand dimensions
    let tensor = tf.browser.fromPixels(img).resizeBilinear([224, 224]);
    tensor = tensor.toFloat().div(tf.scalar(255)).expandDims();
    return tensor as tf.Tensor4D;
  }
}


========================================





import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { Image } from 'image-js';

// Helper function to load images and resize them
async function loadImage(imagePath: string) {
    const img = await Image.load(imagePath);
    const resized = img.resize({ width: 224, height: 224 });
    const tensor = tf.browser.fromPixels(resized.toCanvas()).toFloat();
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
