# **Machine Learning Project Report - Haris Yafie**

Proyek untuk pembelajaran saya dalam kursus Dicoding Machine Learning Terapan. Proyek ini adalah proyek sistem rekomendasi film. Data diperoleh dari Kaggle.com yang berjudul The Movie Dataset.

ID Dicoding: harisyafie

Email: yafie345@gmail.com

# Table of Contents

- [Movie Recommender System](#movie-recommender-system)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
  - [Data Collecting and Loading](#data-collecting-and-loading)
  - [Data Checking](#data-checking)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Statistika Deskriptif](#statistika-deskriptif)
    - [Data Visualization](#data-visualization)
- [Data Preparation](#data-preparation)
  - [Import Library Data Preparation](#import-library-data-preparation)
  - [Data Cleaning](#data-cleaning)
  - [Data Scaling](#data-scaling)
  - [Sequence Generation (Windowing)](#sequence-generation-windowing)
  - [Data Train-Test Splitting](#data-train-test-splitting)
- [Model Development](#model-development)
  - [1. GRU Neural Network](#1-gru-neural-network)
  - [2. LSTM Neural Network](#2-lstm-neural-network)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
  - [Interpretasi Hasil Evaluasi Model GRU](#interpretasi-hasil-evaluasi-model-gru)
  - [Interpretasi Hasil Evaluasi Model LSTM](#interpretasi-hasil-evaluasi-model-lstm)
  - [Kesimpulan Evaluasi Model](#kesimpulan-evaluasi-model)
- [Prediction](#prediction)
  - [Prediksi Harga Emas 30 Hari ke Depan](#prediksi-harga-emas-30-hari-ke-depan)
    - [Grafik Prediksi Harga Emas](#grafik-prediksi-harga-emas)
    - [Forecast Data Preview](#forecast-data-preview)
    - [Interpretasi Tabel Hasil Prediksi](#interpretasi-tabel-hasil-prediksi)
- [Kesimpulan](#kesimpulan)


## Movie Recommender System



---

### Mengapa Prediksi Harga Emas Penting?



**Referensi:**  




## Business Understanding

### Problem Statements
1.

### Goals
1.

### Solution Statements
- 

## Data Understanding

Sebelum membangun model prediksi, penting untuk memahami terlebih dahulu karakteristik dataset yang digunakan. Pada tahap *data understanding*, dilakukan eksplorasi terhadap struktur data, kondisi kualitas data, serta pemahaman terhadap fitur-fitur yang tersedia.

Langkah ini bertujuan untuk memastikan bahwa data yang digunakan benar-benar representatif, relevan, dan siap untuk diproses lebih lanjut dalam tahap modeling. Selain itu, melalui pemahaman awal terhadap data, potensi masalah seperti missing values, atau duplikasi dapat diidentifikasi dan ditangani dengan tepat.

### Data Collecting and Loading

Data yang digunakan dalam proyek ini diperoleh dari Investing.com, dan dapat diakses melalui tautan berikut:  
Kaggle [The Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data).


Dataset yang digunakan mencakup periode **24 April 2023 hingga 22 April 2025**, dengan detail sebagai berikut:



- **Jumlah Data**:  
  - **527 baris** (record harian)  
  - **7 kolom** (fitur)
 
- **Fitur pada Dataset**:
  - **Tanggal**: Tanggal pengamatan harga emas.
  - **Terakhir**: Harga penutupan emas pada hari tersebut (IDR).
  - **Pembukaan (Open)**: Harga pembukaan emas di hari tersebut.
  - **Tertinggi (High)**: Harga tertinggi emas yang tercapai dalam sehari.
  - **Terendah (Low)**: Harga terendah emas yang tercapai dalam sehari.
  - **Vol. (Volume)**: Volume perdagangan emas dalam satuan transaksi.
  - **Perubahan% (Change%)**: Persentase perubahan harga emas dibandingkan dengan hari sebelumnya.

### Data Checking



### **Exploratory Data Analysis**:
   #### **Statistika Deskriptif**



#### **Data Visualization**



## Data Preparation



### Data Cleaning
 

   
### Data Scaling


   
### Sequence Generation (Windowing)


### Data Train-Test Splitting


## Model Development


### 1. GRU Neural Network


### 2. LSTM Neural Network


## Model Training


## Model Evaluation


### **Interpretasi Hasil Evaluasi Model GRU**


### **Interpretasi Hasil Evaluasi Model LSTM**


### **Kesimpulan Evaluasi Model**

## **Prediction**


### **Prediksi Harga Emas 30 Hari ke Depan**


### Grafik Prediksi Harga Emas



### **Interpretasi Grafik Prediksi**


#### 30-Day Gold Price Forecast


## Kesimpulan





