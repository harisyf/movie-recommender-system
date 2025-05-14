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

Di era digital saat ini, pengguna layanan streaming film seperti Netflix, Disney+, maupun platform lokal dihadapkan pada ribuan pilihan film yang tersedia. Jumlah film yang sangat banyak ini justru menjadi tantangan baru — pengguna sering mengalami kebingungan dalam memilih film yang sesuai dengan selera mereka.

Sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menemukan film yang relevan, menarik, dan sesuai dengan preferensi personal. Selain meningkatkan pengalaman pengguna (user experience), sistem ini juga dapat meningkatkan tingkat keterlibatan pengguna (user engagement) terhadap platform.

Proyek ini memiliki dua manfaat utama. Pertama, dari sisi pengguna, sistem ini bertujuan memberikan kemudahan dalam pemilihan film yang sesuai dengan karakteristik atau histori interaksi mereka. Kedua, dari sisi pengembang (penulis), proyek ini menjadi ajang pembelajaran untuk memahami dan menerapkan berbagai pendekatan sistem rekomendasi dalam konteks nyata.

Beberapa pendekatan yang digunakan dalam sistem rekomendasi ini mengacu pada literatur yang telah ada, seperti content-based filtering, collaborative filtering, dan hybrid approach (Badriyah, Restuningtyas, & Setyorini, 2017); (Wijaya & Alfian, 2018); (Prasetya, 2017).

---

### Mengapa Prediksi Harga Emas Penting?

**Sistem rekomendasi sangat penting dalam konteks film** karena keputusan untuk menonton sering kali bersifat impulsif dan dipengaruhi oleh preferensi personal yang sulit didefinisikan secara eksplisit. Tidak seperti produk fisik, film memiliki nilai subjektif yang tinggi — apa yang disukai oleh satu pengguna bisa sangat berbeda dengan yang lain. Dengan adanya sistem rekomendasi, pengguna dapat menerima saran film yang sesuai dengan selera mereka tanpa harus mencari secara manual, sehingga dapat menghemat waktu dan meningkatkan kepuasan dalam menggunakan platform.


**Referensi:**  

- Badriyah, T., Restuningtyas, I., & Setyorini, F. (2017). Sistem Rekomendasi Collaborative Filtering Berbasis User Algoritma Adjusted Cosine Similarity. *Prosiding Seminar Nasional Sisfotek Volume 10 Nomor 1*, 38–45.
- Wijaya, A. E., & Alfian, D. (2018). Sistem Rekomendasi Laptop Menggunakan Collaborative Filtering Dan Content-Based Filtering. *Jurnal Computech & Bisnis*, 14–16.
- Prasetya, C. S. (2017). Sistem Rekomendasi pada E-Commerce Menggunakan K-Nearest Neighbor. *Jurnal Teknologi Informasi dan Ilmu Komputer (JTIIK)* Vol. 4, No. 3, 194-200.


## Business Understanding

### Problem Statement
Pengguna platform streaming film sering kali mengalami kesulitan dalam memilih film yang sesuai dengan preferensi pribadi mereka. Hal ini disebabkan oleh banyaknya pilihan film yang tersedia dan kurangnya sistem yang dapat secara otomatis menyesuaikan rekomendasi berdasarkan karakteristik atau histori pengguna.

### Goal
Tujuan dari proyek ini adalah membangun sistem rekomendasi personalized yang mampu memberikan **10 film teratas (Top-N Recommendation)** sesuai dengan minat pengguna berdasarkan pendekatan yang relevan.

### Solution Approach
Proyek ini mengusulkan solusi berupa sistem rekomendasi **hybrid**, yaitu menggabungkan dua pendekatan utama:
- **Content-Based Filtering (CBF)**: Memanfaatkan informasi konten film seperti judul dan genre untuk mencari kesamaan antar film. Pendekatan ini sangat cocok digunakan pada kasus cold-start, yaitu ketika pengguna belum memiliki histori interaksi yang cukup.
- **Collaborative Filtering (CF)**: Menggunakan data rating antar pengguna untuk mengidentifikasi kesamaan preferensi. Pendekatan ini lebih akurat ketika data interaksi pengguna sudah mencukupi, namun memiliki keterbatasan pada pengguna baru (cold-start problem).
  
Dengan mengombinasikan kedua pendekatan tersebut, sistem rekomendasi diharapkan dapat mengatasi keterbatasan masing-masing metode dan memberikan hasil rekomendasi yang lebih personal dan relevan.

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





