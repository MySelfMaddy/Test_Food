import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { CameraPreview } from '@capacitor-community/camera-preview';
import { ModelServiceService } from './model-service.service';
@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage {
  scanning: boolean = false;
  resultMessage: string = '';
  scanInterval: any;
  private model: any;

  constructor(
    private http: HttpClient,
    private modelService: ModelServiceService
  ) {
    setTimeout(() => {
      this.loadModel();
    }, 1000);
  }
  async loadModel() {
    try {
      this.model = await tf.loadLayersModel('assets/product_quality_model/model.json');
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
    }
  }

  async toggleScanning() {
    this.scanning = !this.scanning;
    if (this.scanning) {
      await this.startCamera();
      this.startScanning();
    } else {
      await this.stopCamera();
    }
  }

  async startCamera() {
    await CameraPreview.start({
      position: 'rear',
      parent: 'cameraPreview',
      toBack: false,
      width: window.innerWidth,
      height: window.innerHeight * 0.7,
    });
  }

  async stopCamera() {
    await CameraPreview.stop();
    clearInterval(this.scanInterval);
  }

  startScanning() {
    this.scanInterval = setInterval(async () => {
      const image = await CameraPreview.capture({ quality: 85 });
      this.stopCamera();
      this.scanning = false;
      this.analyzeImage(image.value);
    }, 3000); // Capture every 3 seconds
  }

  async analyzeImage(base64Image: string) {
    //this.loadModel();
    const img = new Image();
    img.src = `data:image/jpeg;base64,${base64Image}`;
    img.onload = async () => {
      const tensor = tf.browser
        .fromPixels(img)
        .resizeBilinear([128, 128])
        .expandDims();
      const prediction = this.model.predict(tensor);
      const result = await prediction.data();
      this.resultMessage =
        result[0] > 0.5
          ? 'The food item appears to be spoiled.'
          : 'The food item appears fresh!';
    };
  }

  onFileChange(event: Event) {
    this.modelService.handleFileInputChange(event);
  }
}
