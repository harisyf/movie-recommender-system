# **Machine Learning Project Report - Haris Yafie**

Di tengah melimpahnya jumlah film yang tersedia di platform streaming digital, pengguna sering mengalami **kesulitan dalam memilih film** yang sesuai dengan selera mereka. Untuk menjawab tantangan tersebut, proyek ini membangun sebuah **Sistem Rekomendasi Film** berbasis machine learning yang bertujuan memberikan saran film yang **lebih personal dan relevan**. Proyek ini untuk pembelajaran saya dalam kursus Dicoding Machine Learning Terapan. Proyek ini adalah proyek sistem rekomendasi film. Data diperoleh dari Kaggle.com yang berjudul The Movie Dataset.

Proyek ini mengeksplorasi dua pendekatan utama:

1. **Content-Based Filtering (CBF)**  
   Memberikan rekomendasi berdasarkan kemiripan konten antar film, seperti judul, genre, dan kata kunci. Cocok digunakan untuk user baru yang belum banyak memiliki histori rating (*cold-start problem*).

2. **Collaborative Filtering (CF)**  
   Memberikan rekomendasi berdasarkan pola interaksi user terhadap film, khususnya dari data rating. Terdapat dua model yang dibangun:
   - **Memory-Based CF**: menggunakan cosine similarity antar item
   - **RecommenderNet**: model neural network sederhana berbasis embedding

**Project Highlight**
- Melakukan **preprocessing dan rekayasa fitur** dari data metadata film, rating pengguna, dan keywords.
- Membangun beberapa model rekomendasi dan mengevaluasi performanya menggunakan **RMSE** serta interpretasi kualitas rekomendasi.
- Menunjukkan bahwa **CBF efektif dalam menemukan film yang mirip secara tema**, sedangkan **Memory-Based CF unggul dalam memberikan rekomendasi yang personalized**.
- **RecommenderNet** memiliki potensi untuk dikembangkan lebih lanjut, meskipun pada kondisi saat ini performanya masih perlu ditingkatkan.

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

### Mengapa Sistem Rekomendasi Penting?

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

Proyek ini mengembangkan dua pendekatan sistem rekomendasi secara terpisah, yaitu **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)**. Masing-masing pendekatan dirancang untuk memberikan rekomendasi film yang relevan berdasarkan perspektif yang berbeda.

- **Content-Based Filtering (CBF)**  
  Pendekatan ini memanfaatkan informasi konten dari film seperti judul dan genre untuk mengukur kemiripan antar film. Sistem akan merekomendasikan film yang memiliki karakteristik serupa dengan film yang sebelumnya disukai oleh pengguna. Metode ini sangat efektif untuk mengatasi masalah cold-start, yaitu ketika pengguna belum memiliki cukup riwayat interaksi.

- **Collaborative Filtering (CF)**  
  Pendekatan ini didasarkan pada interaksi pengguna, khususnya melalui data rating. Sistem akan merekomendasikan film berdasarkan pola kesamaan preferensi antara pengguna yang berbeda. CF dapat memberikan hasil rekomendasi yang lebih personal jika data interaksi pengguna tersedia dalam jumlah cukup. Namun, metode ini memiliki kelemahan ketika menangani pengguna baru atau item yang belum pernah dirating (cold-start problem).

Kedua pendekatan ini akan dibandingkan untuk mengevaluasi efektivitas masing-masing dalam memberikan rekomendasi top-10 film kepada pengguna.


## 🔍 Data Understanding

Sebelum membangun model prediksi, penting untuk memahami terlebih dahulu karakteristik dataset yang digunakan. Pada tahap *data understanding*, dilakukan eksplorasi terhadap struktur data, kondisi kualitas data, serta pemahaman terhadap fitur-fitur yang tersedia.

Langkah ini bertujuan untuk memastikan bahwa data yang digunakan benar-benar representatif, relevan, dan siap untuk diproses lebih lanjut dalam tahap modeling. Selain itu, melalui pemahaman awal terhadap data, potensi masalah seperti missing values, atau duplikasi dapat diidentifikasi dan ditangani dengan tepat.

### 📥 Data Collecting and Loading

Data yang digunakan dalam proyek ini diperoleh dari Kaggle.com, dan dapat diakses melalui tautan berikut:  
Kaggle [The Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data).

Dataset yang digunakan terdiri dari beberapa file CSV yang saling terkait:

1. `movies_metadata.csv`  
   Berisi informasi metadata dari film seperti judul, genre, tanggal rilis, dan deskripsi (overview).

![Movies Metadata](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/movies_metadata.png)


2. `ratings_small.csv`  
   Berisi interaksi pengguna dalam bentuk rating terhadap film, dengan kolom `userId`, `movieId`, `rating`, dan `timestamp`. Dataset ini digunakan untuk pendekatan Collaborative Filtering.

   ![Rating](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/rating_dataset.png)

3. `keywords.csv`  
   Berisi daftar kata kunci (keywords) untuk setiap film yang dapat digunakan untuk memperkaya fitur konten pada pendekatan Content-Based Filtering.

   ![Keywords](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/keywords_dataset.png)

4. `links_small.csv`  
   Dataset ini menghubungkan `movieId` dari MovieLens ke ID versi TMDb (`tmdbId`), yang diperlukan untuk mencocokkan data antar file.

![Link Dataset](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/link_dataset.png)

Seluruh file telah dimuat ke dalam lingkungan kerja menggunakan library Python seperti `pandas`, dan akan diproses lebih lanjut pada tahap berikutnya. Berikut contoh cara memuat file menggunakan pandas:

```python
import pandas as pd

movies_metadata = pd.read_csv('movies_metadata.csv')
ratings = pd.read_csv('ratings_small.csv')
keywords = pd.read_csv('keywords.csv')
links = pd.read_csv('links_small.csv')
```

### 🔬 Data Checking


Sebelum masuk ke tahap eksplorasi dan pembersihan data lebih lanjut, dilakukan pengecekan awal terhadap masing-masing dataset yang digunakan. Pengecekan ini bertujuan untuk memahami kondisi awal data seperti ukuran (dimensi), keberadaan duplikasi, serta nilai yang hilang (missing values) pada setiap dataset.

Adapun langkah-langkah pengecekan yang dilakukan adalah sebagai berikut:

1. **Nama dan Ukuran Dataset**  
   Menampilkan jumlah baris dan kolom untuk setiap dataset (`movies_metadata`, `ratings`, `keywords`, dan `links`) guna memberikan gambaran awal tentang skala data yang akan diolah.

2. **Pengecekan Duplikasi**  
   Menghitung jumlah baris yang duplikat pada setiap dataset. Duplikasi data dapat memengaruhi kualitas hasil model jika tidak ditangani dengan benar, terutama pada dataset `ratings` dan `movies_metadata`.

3. **Pengecekan Missing Values**  
   Mengidentifikasi kolom-kolom yang memiliki nilai kosong (missing/null). Kolom dengan missing values perlu diperhatikan lebih lanjut karena dapat memengaruhi proses transformasi data dan pelatihan model.

Contoh implementasi kode Python yang digunakan dalam tahap ini:

```python
# Dictionary for results
datasets = {
    "movies_metadata": movies_metadata,
    "ratings": ratings,
    "links": links,
    "keywords": keywords
}

# Loop and Print Data Checking Results
for name, df in datasets.items():
    print(f"Dataset: {name}")
    print(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"- Duplicated Rows: {df.duplicated().sum()}")
    print("\nMissing Values:")
    display(pd.DataFrame(df.isnull().sum(), columns=["Missing Count"]).query("`Missing Count` > 0"))
    print("-" * 50)
```

**Interpretasi Hasil Data Checking**

`movies_metadata`
- Ukuran dataset besar (45K+ baris, 24 kolom), tapi ada **13 baris duplikat** yang perlu dihapus.
- Banyak **missing values** di beberapa kolom:
  - `belongs_to_collection`, `homepage`, dan `tagline` punya missing values yang sangat tinggi (>50%) → pertimbangkan untuk di-drop atau diabaikan tergantung relevansi.
  - Kolom penting seperti `release_date`, `runtime`, dan `overview` juga punya missing → perlu dicek apakah bisa diimputasi atau dibuang.
- Kolom numerik seperti `popularity`, `revenue`, `vote_average`, dll memiliki sedikit missing → bisa diimputasi (mean/median) atau drop.

`ratings`
- Tidak ada missing values maupun duplikat.
- Dataset bersih dan siap diproses.

`links`
- Tidak ada duplikat.
- Terdapat **13 missing values** pada kolom `tmdbId` → bisa dicek kembali relevansinya sebelum dibuang.

`keywords`
- Ada **987 baris duplikat** → perlu dihapus untuk mencegah data redundancy.
- Tidak ada missing values.


### 📊 **Exploratory Data Analysis**:


EDA dilakukan untuk memahami struktur dan karakteristik awal dari setiap dataset.  
Tahapan ini penting untuk menemukan pola, insight, dan potensi masalah sebelum masuk ke tahap pemodelan.

EDA dilakukan secara terpisah untuk tiap dataset sebagai berikut:

---

1. `Movies Metadata` Dataset
- **Visualisasi Genre Populer:**  
  Menggunakan bar chart untuk menampilkan genre film yang paling sering muncul.

2. `Rating` Dataset
- **Statistik Deskriptif:**  
  Menghitung nilai minimum, maksimum, mean, dan distribusi rating.
- **Visualisasi Distribusi Rating:**  
  Membuat histogram untuk melihat persebaran nilai rating yang diberikan oleh user.

3. `Links` Dataset
- **Analisis ID Unik:**  
  Mengecek keberagaman dan kelengkapan ID film dari berbagai sumber (IMDb, TMDb, dll).

4. `Keywords` Dataset
- **Top 20 Keyword Terpopuler:**  
  Menampilkan 20 keyword yang paling sering digunakan.
- **Visualisasi Word Cloud:**  
  Membuat visualisasi cloud untuk menggambarkan frekuensi kata secara visual dan menarik.

---

EDA ini bertujuan memberikan insight awal untuk membantu proses data preparation dan pemodelan sistem rekomendasi film ke depannya.

**1. Movies Metadata Dataset**
**Visualisasi Genre Populer:**  
  Menggunakan bar chart untuk menampilkan genre film yang paling sering muncul. Langkah ini bertujuan untuk mengetahui genre film yang paling sering muncul dalam dataset.

- Data genre diekstrak dari kolom `genres` yang berupa string list dictionary.
- Hanya genre yang masuk dalam whitelist TMDB (`VALID_GENRES`) yang dihitung.
- Nilai yang tidak valid atau tidak bisa diparse akan diabaikan.
- Setelah diekstrak, list genre di-*explode* agar satu baris berisi satu genre.
- Visualisasi dilakukan dengan bar chart untuk melihat distribusi genre film yang paling umum.

Hasilnya menunjukkan genre yang paling sering dipakai dalam metadata film

![Popular Genre Barchart](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/movies_popular_genre.png)

**Interpretasi Visualisasi Genre Film Terpopuler**

Dari hasil visualisasi genre film pada dataset `movies_metadata`, terlihat bahwa:

- **Drama** merupakan genre yang paling dominan, dengan jumlah film terbanyak, diikuti oleh **Comedy**, **Thriller**, dan **Romance**.
- Genre-genre populer tersebut cenderung mencerminkan preferensi umum industri film untuk tema-tema emosional, hiburan ringan, dan ketegangan.
- Genre seperti **Western**, **TV Movie**, dan **War** memiliki jumlah film yang relatif sedikit, menandakan bahwa genre ini lebih niche atau kurang diproduksi secara massal.
- Distribusi genre menunjukkan adanya konsentrasi pada genre-genre mainstream, sementara genre lain cenderung memiliki representasi yang jauh lebih kecil.

Insight ini dapat membantu dalam memahami tren dominan dalam industri film, serta berguna dalam pengembangan sistem rekomendasi berdasarkan genre populer.


**2. Rating Dataset**
- **Statistik Deskriptif:**  
  Menghitung nilai minimum, maksimum, mean, dan distribusi rating.
  
Berikut adalah statistik deskriptif dari kolom `rating` yang diberikan oleh pengguna terhadap film:

| Statistik | Nilai       |
|-----------|-------------|
| Count     | 100,004     |
| Mean      | 3.5436      |
| Std       | 1.0581      |
| Min       | 0.5         |
| 25%       | 3.0         |
| 50%       | 4.0         |
| 75%       | 4.0         |
| Max       | 5.0         |

- **Interpretasi Statistik Deskriptif `rating`**

Berdasarkan hasil statistik deskriptif:

- Jumlah data rating: **100.004** entri.
- **Rata-rata (mean)** rating: sekitar **3.54**, menunjukkan kecenderungan user memberikan rating cukup tinggi secara umum.
- **Median (50%)** dan **kuartil 75%** bernilai **4.0**, artinya lebih dari setengah user memberi rating ≥ 4.
- **Kuartil 25%** bernilai **3.0**, menunjukkan bahwa 75% rating berada di atas nilai ini.
- Nilai rating **berkisar antara 0.5 hingga 5.0**, dengan **standar deviasi sebesar 1.05**, mengindikasikan bahwa persebaran rating cukup terkonsentrasi di sekitar nilai tengah (3–4).

**Insight:**  
User cenderung memberikan rating yang positif terhadap film yang mereka tonton. Hal ini bisa berpengaruh pada sistem rekomendasi, karena adanya bias positif dalam persebaran rating.


- **Visualisasi Distribusi Rating:**  
  Membuat histogram untuk melihat persebaran nilai rating yang diberikan oleh user.

  ![Distribusi Rating](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/distribution_rating.png)

**Interpretasi Histogram Rating Film**

Visualisasi menunjukkan distribusi diskret rating yang diberikan user terhadap film:

- Rating **4.0** adalah yang paling sering diberikan, diikuti oleh **3.0** dan **4.5**.
- Mayoritas rating berada pada rentang **3.0 hingga 5.0**, menandakan kecenderungan user memberi rating yang cukup tinggi.
- Rating rendah seperti **0.5 hingga 2.0** jauh lebih jarang muncul, menunjukkan bahwa user cenderung jarang memberi nilai jelek.
- Distribusi bersifat **right-skewed** (condong ke kanan), mencerminkan adanya **bias positif** dalam penilaian pengguna terhadap film.

**Insight:**  
Kecenderungan rating yang tinggi ini perlu diperhatikan dalam model rekomendasi, karena bisa menyebabkan overfitting terhadap film yang populer atau banyak dinilai positif.


**3. Links Dataset**
- **Analisis ID Unik:**  
  Mengecek keberagaman dan kelengkapan ID film dari berbagai sumber (IMDb, TMDb, dll).
Berikut adalah ringkasan informasi terkait jumlah ID unik dan nilai yang hilang pada dataset `links_small.csv`:

| Kolom       | Jumlah Unik |
|-------------|-------------|
| `movieId`   | 9,125       |
| `imdbId`    | 9,125       |
| `tmdbId`    | 9,112       |

| Kolom       | Jumlah Missing |
|-------------|----------------|
| `tmdbId`    | 13             |

**Interpretasi EDA `links` Dataset**

Berdasarkan hasil eksplorasi:

- Jumlah nilai unik untuk `movieId` dan `imdbId` adalah **9125**, menandakan satu-ke-satu mapping yang konsisten di antara keduanya.
- `tmdbId` memiliki **9112 nilai unik**, sedikit lebih rendah dibanding yang lain.
- Terdapat **13 nilai `tmdbId` yang hilang (missing)**, sesuai dengan hasil data checking sebelumnya.
- Tipe data untuk `tmdbId` adalah `float64`, kemungkinan karena adanya nilai `NaN` (missing) yang membuatnya tidak terbaca sebagai `int64`.

**Insight:**  
`links` dataset memiliki struktur yang cukup rapi untuk menghubungkan berbagai sumber ID film.  
Namun, **13 baris dengan `tmdbId` yang hilang perlu ditangani** sebelum dilakukan merge atau pemodelan, terutama jika data dari TMDb digunakan sebagai referensi utama.


#### **4. Keywords Dataset**
Langkah ini bertujuan untuk menganalisis kata kunci (`keywords`) yang digunakan untuk mendeskripsikan film dalam dataset.

**Proses**:
- Kolom `keywords` berisi string list dari dictionary, sehingga perlu diekstrak menjadi list kata kunci menggunakan fungsi `extract_keywords`.
- Data kemudian di-*explode* agar setiap baris hanya berisi satu keyword untuk mempermudah analisis frekuensi.
- Dilakukan dua bentuk visualisasi:
  1. **Bar chart** untuk menampilkan **20 keyword paling umum** yang paling sering muncul.
  2. **Word cloud** untuk menyajikan visualisasi frekuensi keyword dalam bentuk artistik dan intuitif.
 

**Top 20 Keywords Film**

![Top 20 Keywords](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/top_20_keywords.png)

**Interpretasi Grafik Top 20 Keywords Film**

Grafik menunjukkan 20 kata kunci (`keywords`) yang paling sering muncul dalam deskripsi film:

- **"woman director"** menempati posisi teratas dengan lebih dari 3.000 film, menandakan bahwa pencantuman gender sutradara sebagai keyword cukup umum dilakukan — bisa jadi bagian dari tag identitas atau representasi.
- Keyword seperti **"independent film"**, **"murder"**, dan **"based on novel"** menggambarkan karakteristik umum dari produksi atau cerita film.
- Banyak keyword yang berkaitan dengan tema konten dewasa atau emosional seperti **"sex"**, **"violence"**, **"nudity"**, dan **"revenge"**, yang bisa menunjukkan kecenderungan film untuk menonjolkan sisi konflik dan intensitas.
- Keyword seperti **"love"**, **"friendship"**, dan **"teenager"** juga muncul, memperkuat bahwa relasi dan dinamika usia adalah tema-tema yang sering dibawa.
- Kehadiran keyword seperti **"sequel"** dan **"duringcreditsstinger"** juga menandakan banyaknya film berformat waralaba atau cinematic universe.

**Insight:**  
Keyword memberi gambaran yang cukup kuat terhadap **tema, tone, dan gaya produksi** dari film. Hal ini sangat berguna sebagai fitur konten dalam sistem rekomendasi berbasis content, terutama ketika digabungkan dengan metadata lain seperti genre dan overview.

**Wordcloud**

![Wordcloud](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/wordcloud.png)

**Interpretasi Word Cloud dari Keywords Film**

Word cloud memberikan gambaran visual terhadap kata kunci yang paling sering digunakan dalam deskripsi film:

- Keyword yang paling menonjol seperti **"woman director"**, **"independent film"**, **"murder"**, dan **"based on novel"** muncul dengan ukuran huruf paling besar, menandakan frekuensi kemunculan yang tinggi.
- Banyak kata kunci yang merepresentasikan tema cerita (misalnya: *death*, *love*, *revenge*, *friendship*, *ghost*, *suicide*), tempat (*new york*, *paris*, *york city*), bahkan karakter spesifik (*serial killer*, *teenager*).
- Kata-kata yang berkaitan dengan genre, gaya produksi, dan situasi juga terlihat, seperti *biography*, *high school*, *war*, *nudity*, *martial arts*, dan *escape*.

**Insight:**
- Word cloud ini menunjukkan **keragaman tema dan karakteristik film** dalam dataset.
- Keyword seperti ini sangat berguna dalam sistem rekomendasi berbasis konten (content-based), karena bisa mencerminkan kesamaan makna antar film yang tidak bisa dilihat hanya dari genre atau rating saja.

Word cloud ini juga memudahkan identifikasi **tema-tema dominan dan tren cerita** yang bisa jadi menarik buat segmentasi pengguna tertentu.


## 🦾 **Data Preparation**

Tahap *Data Preparation* bertujuan untuk membersihkan dan mengolah data agar siap digunakan dalam proses modeling sistem rekomendasi. Berdasarkan alur kode, proses ini terdiri dari empat tahap utama:

---

1. Data Cleaning
2. Sanity Check
3. Merge Dataset
4. Final Feature Datasets (CBF & CF)


### 🧹 **Data Cleaning**

Langkah ini fokus pada konsistensi format dan penghapusan data yang tidak valid. Beberapa aksi utama yang dilakukan:

- **Konsistensi Kolom ID:**
  - Kolom `id` pada `movies_metadata` dikonversi menjadi numerik agar cocok dengan `tmdbId` di dataset `links`.
  - Nilai `NaN` pada kolom ID dihapus sebelum konversi tipe data.

- **Penghapusan Duplikat:**
  - Duplikat baris berdasarkan kolom `id` pada `movies_metadata` dihapus dengan menyimpan film dengan `vote_count` tertinggi.
  - Dataset `ratings`, `links`, dan `keywords` juga dibersihkan dari baris duplikat.

- **Pembersihan Kolom `genres`:**
  - Parsing nilai `genres` menjadi list of strings dan hanya menyimpan genre yang valid sesuai whitelist (misalnya: `'Drama'`, `'Comedy'`, dll).

- **Pembersihan Kolom `keywords`:**
  - Parsing data JSON-like menjadi string yang digabung dan di-lowercase agar lebih mudah digunakan pada content-based filtering.

- **Penanganan Missing Values:**
  - Baris yang tidak memiliki `title` dihapus.
  - Kolom-kolom dengan proporsi missing value lebih dari 30% dihapus dari dataset.

---


### 🫧 **Sanity Check**

Tahap ini memastikan bahwa hasil *cleaning* berjalan dengan benar. Beberapa validasi yang dilakukan:

- Memastikan tidak ada ID film yang duplikat.
- Mengecek apakah kolom penting seperti `title` dan `genres` tidak mengandung missing value.
- Validasi tipe data (`genres` bertipe list dan `keywords` bertipe string).
- Memastikan bahwa semua genre berada dalam daftar whitelist yang ditentukan.

Jika ada keanehan seperti tipe data tidak sesuai atau genre tak dikenal, proses akan dihentikan melalui `assert`.


### 🧩 **Merge Datasets**

Setelah data bersih, dilakukan proses penggabungan dataset agar siap untuk modeling:

- `movies_metadata` digabung dengan `keywords` berdasarkan kolom `id`.
- Hasilnya kemudian digabung lagi dengan `links` untuk mendapatkan kolom `movieId` sebagai key utama yang digunakan baik untuk CB maupun CF.
- Kolom `tmdbId` dihapus karena `movieId` akan digunakan sebagai *primary key* universal.
- Dataset `ratings` tetap dipisah karena hanya digunakan pada collaborative filtering.

---


### **Final Feature Dataset**

Dua dataset akhir disiapkan berdasarkan pendekatan yang digunakan:

- **Content-Based Filtering (CB):**
  - Disusun dari `movies_metadata`, `keywords`, dan `links`, menghasilkan dataframe `cb_movies`.
  - Fitur yang digunakan mencakup `title`, `genres`, `keywords`, `popularity`, `vote_average`, `vote_count`, dan `release_year` (diambil dari `release_date`).

- **Collaborative Filtering (CF):**
  - Menggunakan user-item rating matrix dari dataset `ratings`, menghasilkan dataframe `cf_ratings` berisi `userId`, `movieId`, dan `rating`.

Kedua dataframe ini akan digunakan sebagai basis untuk membangun sistem rekomendasi berbasis konten dan kolaboratif.

Content Based Feature Dataset: (9082, 10)

![CBF Dataset](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/cb_dataset.png)


Collaborative Filtering Feature Dataset: (100004, 3)

![CF Dataset](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/cf_ratings_dataset.png)

## ⚙️ **Model Development**

Pada tahap ini, kita membangun dan mengembangkan dua pendekatan utama dalam sistem rekomendasi film, yaitu **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)**. Masing-masing pendekatan dikembangkan melalui tahapan yang sistematis mulai dari *model building*, *training*, hingga *evaluation* untuk mengukur performanya.

- **Content-Based Filtering** merekomendasikan film berdasarkan kemiripan konten seperti genre, judul, dan kata kunci dari film yang pernah disukai user.
- **Collaborative Filtering** fokus pada pola interaksi user terhadap film, yaitu menggunakan data rating sebagai dasar untuk memprediksi preferensi.

Selain membangun model, tahap ini juga mencakup proses evaluasi untuk membandingkan performa antar pendekatan. Hasil evaluasi akan menjadi dasar dalam menentukan model mana


### **Content Based Filtering**

Content-Based Filtering (CBF) adalah pendekatan rekomendasi yang berfokus pada karakteristik konten dari setiap item. Dalam konteks ini, kita menggunakan informasi dari film seperti **judul**, **kata kunci (keywords)**, dan **genre** untuk menentukan kemiripan antar film. Ide dasarnya: jika user menyukai sebuah film, maka mereka kemungkinan juga akan menyukai film lain yang memiliki konten serupa.

Pendekatan ini tidak memerlukan data dari user lain dan bisa bekerja dengan baik bahkan ketika user baru hanya menyukai satu atau dua film — cocok untuk mengatasi masalah *cold-start* pada user. Di bagian ini, kita akan membangun dua model content-based dengan teknik vektorisasi berbeda, lalu membandingkan hasil rekomendasinya.

#### **Model Building**

Pada tahap ini, model content-based dibangun dengan cara menggabungkan beberapa fitur penting dari setiap film, yaitu:
- **Judul film (`title`)**
- **Kata kunci (`keywords`)**
- **Genre (`genres`)**

Fitur-fitur tersebut digabungkan ke dalam satu kolom teks `combined_features`, yang nantinya akan digunakan sebagai dasar dalam proses perhitungan kesamaan antar film.

Langkah-langkah utama dalam pembangunan model:
1. **Preprocessing**: Menggabungkan dan menormalkan teks dari kolom fitur (diubah ke lowercase dan digabung jadi satu string).
2. **Vectorization**: Mengubah teks menjadi representasi numerik menggunakan dua teknik berbeda:
   - `TF-IDF Vectorizer` untuk menangkap bobot pentingnya kata.
   - `CountVectorizer` untuk menghitung frekuensi kata.
3. **Similarity Computation**: Menghitung **cosine similarity** antar semua film berdasarkan hasil vektorisasi.
4. **Rekomendasi**: Dibuat fungsi `get_recs()` yang akan mengembalikan Top-N film yang paling mirip dengan film input berdasarkan skor similarity-nya.

Model yang dihasilkan bersifat fleksibel karena dapat diganti-ganti jenis vectorizer-nya sesuai kebutuhan. Proses ini menghasilkan sistem rekomendasi berbasis konten yang tidak bergantung pada data dari user lain.


#### **Model Computation**

Setelah proses pembangunan model disiapkan, tahap selanjutnya adalah melakukan komputasi vektor dan matriks kesamaan antar film. Di bagian ini, dua model content-based dikonstruksi menggunakan pendekatan berbeda untuk vectorization:

- **TF-IDF Vectorizer**  
  Mengubah teks menjadi vektor berdasarkan *Term Frequency–Inverse Document Frequency*, sehingga kata-kata yang umum diabaikan dan kata-kata unik tiap film diberi bobot lebih tinggi.  
  Selain itu, digunakan `ngram_range=(1, 2)` untuk mempertimbangkan unigram dan bigram.

- **CountVectorizer**  
  Mengubah teks menjadi representasi vektor sederhana berdasarkan jumlah kemunculan kata (frekuensi), tanpa mempertimbangkan bobot pentingnya.

Masing-masing vectorizer dimasukkan ke dalam fungsi `_build_cb_model()` untuk menghasilkan dua model rekomendasi:
- `get_recs_tfidf` → model berbasis TF-IDF
- `get_recs_count` → model berbasis CountVectorizer

Kedua model ini siap digunakan untuk menghitung kesamaan antar film dan menghasilkan rekomendasi berbasis konten yang berbeda gaya pendekatannya.


#### **Model Evaluation**

Evaluasi model Content-Based Filtering dilakukan menggunakan metode sederhana namun efektif yang disebut **sanity check**. Tujuannya adalah untuk melihat apakah model mampu merekomendasikan film yang relevan berdasarkan preferensi user.

Langkah-langkah evaluasinya:
1. **Ambil user sample** (misalnya `user_id = 45`) dan cari film yang mereka beri rating tinggi (≥ 4.0).
2. **Pilih satu film acak** dari daftar film yang disukai tersebut sebagai referensi (`ref_title`).
3. **Gunakan model rekomendasi** untuk mencari Top-N film yang mirip secara konten dengan film referensi.
4. **Bandingkan hasil rekomendasi** dari dua model:
   - TF-IDF + Cosine Similarity
   - CountVectorizer + Cosine Similarity

Hasil evaluasi ditampilkan dalam bentuk tabel yang mencantumkan:
- Film-film yang disukai user
- Rekomendasi film dari masing-masing model
- Genre dan skor similarity dari setiap film yang direkomendasikan

Meskipun sederhana, metode ini cukup efektif untuk memastikan bahwa model mampu menangkap kesamaan konten yang relevan dengan preferensi pengguna.

**Referensi Film dan Rating Tinggi oleh User 45**

**Reference film**: *Beauty and the Beast*  
**Daftar film yang diberi rating ≥ 4 oleh user 45:**

| #  | movieId | Title                   | Genres                             | Rating |
|----|---------|-------------------------|-------------------------------------|--------|
| 1  | 903     | Vertigo                 | Mystery Romance Thriller           | 5.0    |
| 2  | 26151   | Au Hasard Balthazar     | Drama                              | 5.0    |
| 3  | 7064    | Beauty and the Beast    | Drama Fantasy Romance              | 5.0    |
| 4  | 1673    | Boogie Nights           | Drama                              | 4.5    |
| 5  | 1333    | The Birds               | Horror                             | 4.5    |
| 6  | 3307    | City Lights             | Comedy Drama Romance               | 4.5    |
| 7  | 3160    | Magnolia                | Drama                              | 4.5    |
| 8  | 1748    | Dark City               | Mystery Science Fiction            | 4.5    |
| 9  | 2692    | Run Lola Run            | Action Drama Thriller              | 4.5    |
| 10 | 1199    | Brazil                  | Comedy Science Fiction             | 4.0    |
| 11 | 899     | Singin' in the Rain     | Comedy Music Romance               | 4.0    |
| 12 | 5878    | Talk to Her             | Drama Romance                      | 4.0    |
| 13 | 4235    | Amores perros           | Drama Thriller                     | 4.0    |

---

**Top-10 Recommendations – TF-IDF Model**

| #  | movieId  | Title                       | Genres                                         | Similarity |
|----|----------|-----------------------------|------------------------------------------------|------------|
| 1  | 595      | Beauty and the Beast        | Romance Family Animation Fantasy Music         | 0.232116   |
| 2  | 781      | Stealing Beauty             | Drama Romance                                  | 0.169768   |
| 3  | 8915     | Stage Beauty                | Comedy Drama Romance                           | 0.169187   |
| 4  | 2563     | Dangerous Beauty            | Drama Romance                                  | 0.160937   |
| 5  | 100032   | Beauty Is Embarrassing      | Documentary                                    | 0.145942   |
| 6  | 8929     | Black Beauty                | Drama Family                                   | 0.139603   |
| 7  | 8851     | Smile                       | Comedy                                         | 0.126161   |
| 8  | 3612     | The Slipper and the Rose    | Adventure Family Fantasy Romance               | 0.125474   |
| 9  | 46578    | Little Miss Sunshine        | Comedy Drama                                   | 0.125412   |
| 10 | 103984   | The Great Beauty            | Comedy Drama                                   | 0.120418   |

---

**Top-10 Recommendations – CountVectorizer Model**

| #  | movieId  | Title                       | Genres                                         | Similarity |
|----|----------|-----------------------------|------------------------------------------------|------------|
| 1  | 781      | Stealing Beauty             | Drama Romance                                  | 0.577350   |
| 2  | 8915     | Stage Beauty                | Comedy Drama Romance                           | 0.516398   |
| 3  | 2563     | Dangerous Beauty            | Drama Romance                                  | 0.500000   |
| 4  | 595      | Beauty and the Beast        | Romance Family Animation Fantasy Music         | 0.421350   |
| 5  | 8929     | Black Beauty                | Drama Family                                   | 0.387298   |
| 6  | 61361    | The Women                   | Comedy Drama Romance                           | 0.384900   |
| 7  | 58998    | Forgetting Sarah Marshall   | Comedy Romance Drama                           | 0.365148   |
| 8  | 2894     | Romance                     | Romance Drama Comedy                           | 0.353553   |
| 9  | 147010   | Thou Gild'st the Even       | Drama Romance Fantasy                          | 0.353553   |
| 10 | 55498    | Silk                        | Drama Romance                                  | 0.333333   |


**Interpretasi Evaluasi Model**

Berdasarkan hasil evaluasi menggunakan user dengan ID `45`, model merekomendasikan film yang memiliki kemiripan konten dengan salah satu film favorit user, yaitu **"Beauty and the Beast"**.

User ini memiliki preferensi yang cukup jelas terhadap genre **Drama**, **Romance**, dan beberapa elemen **Fantasy**, **Thriller**, dan **Comedy**. Hal ini terlihat dari daftar film yang mereka beri rating tinggi, seperti *Vertigo*, *Boogie Nights*, *City Lights*, dan *Talk to Her*.

**TF-IDF Model**
Rekomendasi dari model TF-IDF menampilkan film-film yang cukup relevan secara tematik dan emosional, seperti:
- *Stealing Beauty*
- *Stage Beauty*
- *Dangerous Beauty*

Meskipun skor similarity relatif kecil (di bawah 0.25), model tetap mampu menyarankan film yang berkaitan erat dengan kata kunci "beauty" dan nuansa romantis. TF-IDF cenderung lebih fokus pada **kata-kata unik dan bobot tematik**.

**CountVectorizer Model**
Model ini memberikan skor similarity yang lebih tinggi secara numerik (hingga > 0.5), dengan film-film yang hampir identik secara kata seperti:
- *Stealing Beauty*
- *Stage Beauty*
- *Dangerous Beauty*

Hasilnya sangat mirip dengan TF-IDF dalam hal judul, tetapi lebih literal karena CountVectorizer hanya menghitung **frekuensi kata**. Model ini lebih mudah "terpengaruh" oleh kemiripan kata secara eksplisit daripada bobot pentingnya.

**Insight**
- Kedua model berhasil menemukan film-film yang berada di **jalur preferensi user**, terutama dalam tema romantis dan drama.
- TF-IDF bekerja baik untuk menangkap nuansa dan bobot kata, meski skor similarity-nya lebih kecil.
- CountVectorizer lebih sensitif terhadap kata yang sering muncul, menghasilkan skor similarity yang lebih tinggi tapi juga lebih "kasar" dalam interpretasi.

Secara keseluruhan, hasil ini menunjukkan bahwa **model content-based dapat memberikan rekomendasi yang relevan** dengan memanfaatkan informasi konten film, bahkan tanpa data interaksi user lain.

**Sehingga model yang digunakan yaitu TF-IDF Vectorized**


### **Collaborative Filtering**

Collaborative Filtering (CF) adalah pendekatan sistem rekomendasi yang berdasarkan pada pola interaksi pengguna terhadap item, bukan pada konten dari item itu sendiri. Dalam konteks proyek ini, CF memanfaatkan **data rating dari user terhadap film** untuk mempelajari pola preferensi yang serupa antar pengguna atau antar film.

Ide utamanya: *"Jika dua user menyukai film yang sama, maka mereka cenderung akan menyukai film lainnya yang juga disukai oleh user dengan preferensi serupa."*

CF sangat efektif untuk menghasilkan rekomendasi **personalized** karena benar-benar didasarkan pada kebiasaan dan interaksi nyata dari pengguna. Namun, pendekatan ini juga memiliki tantangan seperti:
- *Cold-start problem* untuk user baru tanpa histori rating
- *Sparsity problem* karena kebanyakan user hanya memberi rating ke sedikit film

Dalam proyek ini, dua pendekatan CF akan digunakan:
1. **Memory-Based Collaborative Filtering** – Menghitung kesamaan antar item berdasarkan user-item matrix.
2. **Model-Based Collaborative Filtering** dengan neural network sederhana (*RecommenderNet*).

Keduanya akan dibandingkan dari sisi hasil rekomendasi dan performa metrik.


#### **Model Building**

Pada Collaborative Filtering, kita membangun dua jenis model berdasarkan data rating antar user dan film:

---

**1. Memory-Based Collaborative Filtering**

Pendekatan ini menggunakan **item-item similarity** berdasarkan pola rating yang diberikan oleh user. Langkah-langkah utamanya:

- Membuat **user-item matrix** dari data rating.
- Mengubahnya menjadi matriks sparse (`csr_matrix`) untuk efisiensi komputasi.
- Menghitung **cosine similarity** antar film berdasarkan rating user.
- Mengembangkan fungsi `get_recs_memory()` untuk memberikan rekomendasi film kepada user berdasarkan rating tinggi dari film serupa yang pernah ditonton.

Pendekatan ini mudah diimplementasikan dan tidak memerlukan proses training model, tetapi bisa terkena masalah sparsity jika banyak film belum memiliki cukup rating.

---

**2. Model-Based Collaborative Filtering (RecommenderNet)**

Model ini menggunakan pendekatan neural network sederhana untuk mempelajari hubungan antara user dan film. Langkah-langkahnya:

- Melakukan encoding terhadap `userId` dan `movieId` menjadi indeks numerik.
- Membagi data menjadi **training dan validation set**.
- Membangun arsitektur **embedding layer** untuk user dan film.
- Menggunakan **dot product** dari embedding sebagai prediksi rating.
- Melatih model menggunakan *Mean Squared Error (MSE)* dan mengevaluasi dengan *Root Mean Squared Error (RMSE)*.

Model ini dikenal sebagai **RecommenderNet** dan mampu menangkap representasi laten dari user dan film, memberikan hasil yang lebih fleksibel dan akurat dalam jangka panjang, terutama untuk dataset yang besar.

---

Kedua model ini akan dibandingkan pada tahap evaluasi untuk melihat mana yang lebih optimal dalam memberikan rekomendasi personalized.


#### **Train the Model**

Proses training dilakukan dengan memanggil dua fungsi utama:

- `build_cf_memory()` untuk membuat model Memory-Based Collaborative Filtering. Fungsi ini membentuk user-item matrix dan menghitung similarity antar item.
- `build_cf_recommendernet()` untuk melatih model neural network sederhana (RecommenderNet) berbasis embedding. Model ini dilatih menggunakan data rating user untuk mempelajari representasi laten dari user dan film.

Kedua model ini selanjutnya siap dievaluasi dan digunakan untuk menghasilkan rekomendasi personalized.


#### **Model Evaluation**

Evaluasi model Collaborative Filtering dilakukan dengan menggunakan metrik **Root Mean Squared Error (RMSE)**, yang mengukur seberapa dekat prediksi model terhadap rating aktual dari user.

Dua model dievaluasi secara terpisah:

- **Memory-Based CF**  
  Prediksi rating dihitung menggunakan kontribusi rating dari film serupa (weighted average by similarity). Nilai RMSE dihitung berdasarkan sampel acak dari data rating.

- **RecommenderNet**  
  RMSE dihitung pada data validasi yang telah dipisahkan sebelumnya. Selain itu, ditampilkan juga **learning curve** untuk memvisualisasikan perkembangan error selama training.

Akhirnya, kedua nilai RMSE dibandingkan secara visual menggunakan bar chart sederhana untuk melihat model mana yang memberikan performa prediksi terbaik.


![rec_net_rmse](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/rec_net_rmse.png)


**Interpretation – RecommenderNet RMSE Curve**

Grafik di atas menunjukkan nilai **Root Mean Squared Error (RMSE)** pada data training dan validation selama 10 epoch pelatihan model RecommenderNet.

Berikut insight yang dapat diperoleh dari grafik tersebut:

- **Penurunan RMSE yang konsisten**: RMSE pada kedua data (train dan val) menurun tajam di awal epoch (0–3), menunjukkan bahwa model berhasil belajar dari data dan memperbaiki kesalahan prediksi secara signifikan pada fase awal pelatihan.

- **Stabil setelah epoch ke-3**: Setelah mencapai epoch ke-3, baik RMSE training maupun validation mulai stabil dan konvergen. Ini menandakan model mulai mendekati kapasitas optimalnya tanpa overfitting.

- **Gap yang kecil antara train dan val**: Jarak antara kurva training dan validation relatif kecil, yang menunjukkan bahwa model **tidak overfit** dan mampu melakukan generalisasi yang baik ke data yang belum pernah dilihat.

Kesimpulannya, model RecommenderNet menunjukkan performa yang baik dan stabil selama training, dan siap digunakan untuk menghasilkan rekomendasi personalized secara akurat.



![rmse_comparison](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/rmse_comparison.png)


**RMSE Comparison – Memory-Based vs RecommenderNet**

Grafik di atas membandingkan performa dua model Collaborative Filtering berdasarkan nilai **Root Mean Squared Error (RMSE)**:

| Model              | RMSE     |
|-------------------|----------|
| Memory-Based       | **0.931** |
| RecommenderNet     | 2.781     |

**Interpretasi:**

- **Memory-Based Collaborative Filtering** memiliki performa yang jauh lebih baik berdasarkan metrik RMSE. Nilai 0.931 menunjukkan bahwa prediksi rating yang dihasilkan cukup dekat dengan rating aktual user.
- **RecommenderNet**, meskipun menggunakan pendekatan neural network yang lebih kompleks, menghasilkan RMSE yang lebih tinggi (2.781), yang mengindikasikan bahwa prediksi ratingnya masih cukup meleset dibandingkan dengan metode memory-based.

**Insight:**

- Performa RecommenderNet kemungkinan besar bisa ditingkatkan dengan:
  - Pelatihan yang lebih panjang (lebih banyak epoch)
  - Peningkatan arsitektur model (lebih dalam atau pakai regularisasi)
  - Normalisasi data rating (misalnya skala ke 0–1)
  - Penggunaan hyperparameter tuning (batch size, learning rate, embedding size)

Namun, untuk kondisi saat ini, **model Memory-Based menjadi pilihan yang lebih baik** dalam hal akurasi prediksi rating.


## **Recommendation Result**

Setelah seluruh model dibangun, dikomputasi, dan dievaluasi, tahap selanjutnya adalah menghasilkan **rekomendasi film Top-N** untuk pengguna. Rekomendasi ini dihasilkan dari dua pendekatan utama:

1. **Content-Based Filtering Recommendation**  
   Memberikan rekomendasi berdasarkan kesamaan konten film (judul, genre, keywords) dengan film yang disukai oleh user.

2. **Collaborative Filtering Recommendation**  
   Memberikan rekomendasi personalized berdasarkan pola interaksi user terhadap film, baik menggunakan kemiripan antar item (memory-based) maupun representasi laten user-item (RecommenderNet).

Output dari masing-masing pendekatan berupa daftar **Top-N film** yang disarankan untuk user tertentu, disertai dengan skor kemiripan atau prediksi rating.

Rekomendasi ini dapat digunakan untuk mengevaluasi kualitas prediksi model secara kualitatif, serta melihat seberapa relevan film yang disarankan terhadap preferensi pengguna.

### **Content Based Filtering Recommendation**

Pada bagian ini, sistem akan memberikan **Top-N rekomendasi film** berdasarkan kemiripan konten dengan film yang menjadi referensi. Pendekatan yang digunakan adalah:

- **Model**: TF-IDF Vectorizer + Cosine Similarity
- **Input**: Judul film yang disukai user
- **Output**: Daftar film yang memiliki kemiripan konten tertinggi dengan film tersebut

Fungsi `recommend_cb()` digunakan untuk:
1. Mengambil film referensi berdasarkan judul input.
2. Menggunakan model content-based untuk mencari film yang mirip secara konten (judul, keyword, genre).
3. Mengembalikan rekomendasi Top-N film dengan skor similarity tertinggi, disertai informasi genre.

Contoh penggunaannya:
```python
recommend_cb("The Matrix")
```

![recommendation_result_cbf](https://raw.githubusercontent.com/harisyf/movie-recommender-system/main/images/recommendation_result_cbf.png)

**Interpretation – Content-Based Recommendation (TF-IDF)**

Rekomendasi di atas dihasilkan oleh model Content-Based Filtering dengan pendekatan **TF-IDF + Cosine Similarity**, berdasarkan film referensi: **"The Matrix"**.

**Insight dari Rekomendasi:**

- Film-film yang direkomendasikan memiliki kemiripan genre yang kuat dengan *The Matrix*, yaitu dominasi elemen **Action**, **Thriller**, dan **Science Fiction**.
- Beberapa judul seperti:
  - *The Matrix Reloaded* dan *The Matrix Revolutions* — adalah sekuel langsung dari film aslinya, sehingga kemiripannya sangat tinggi secara konten.
  - *Terminator 3*, *I, Robot*, dan *Ghost in the Shell* — mengangkat tema futuristik, kecerdasan buatan, dan perlawanan terhadap sistem, selaras dengan nuansa *The Matrix*.
- Genre yang paling sering muncul di daftar adalah **Science Fiction**, menunjukkan bahwa model berhasil menangkap genre inti dari film referensi.

**Skor Similarity:**

- Skor tertinggi dicapai oleh *The Matrix Reloaded* (0.359), diikuti *The Matrix Revolutions* (0.290), yang masuk akal mengingat keterkaitannya dalam waralaba.
- Skor similarity menurun secara bertahap, menandakan model mampu memprioritaskan film dengan kesamaan konten lebih tinggi terlebih dahulu.

**Kesimpulan:**

Model berhasil memberikan rekomendasi yang **relevan secara tematik dan genre**, dan mampu mengenali hubungan konten baik eksplisit (franchise) maupun implisit (tema dan nuansa). Ini membuktikan bahwa pendekatan content-based dapat memberikan saran yang akurat meskipun hanya bermodal satu film sebagai referensi.


### **Collaborative Filtering Recommendation**

Pada bagian ini, sistem memberikan **Top-N rekomendasi film secara personalized** untuk user tertentu menggunakan pendekatan Collaborative Filtering berbasis memori (item-item).

Fungsi `recommend_cf()` bekerja dengan dua output utama:
1. **Top-N rekomendasi film** untuk user berdasarkan pola rating terhadap film serupa.
2. **Daftar film yang sebelumnya disukai user** (rating ≥ threshold), sebagai perbandingan relevansi.

Model yang digunakan:
- **Memory-Based Collaborative Filtering**
- Pendekatan item-item dengan cosine similarity antar film

Parameter yang digunakan:
- `user_id`: ID pengguna target
- `top_n`: jumlah film yang direkomendasikan
- `threshold`: batas minimal rating yang dianggap “disukai” user (default: 4.0)

Fungsi ini berguna untuk mengevaluasi apakah model berhasil memberikan rekomendasi yang sesuai dengan selera pengguna berdasarkan histori interaksinya.

Contoh penggunaannya:
```python
recommend_cf(user_id=45)
```

| Rank | movieId | Title                   | Genres                                      | Predicted Score |
|------|---------|--------------------------|---------------------------------------------|-----------------|
| 1    | 1252    | Chinatown                | Crime Drama Mystery Thriller                | 19.394881       |
| 2    | 2997    | Being John Malkovich     | Fantasy Drama Comedy                        | 18.961508       |
| 3    | 1617    | L.A. Confidential        | Crime Drama Mystery Thriller                | 18.634464       |
| 4    | 908     | North by Northwest       | Mystery Thriller                            | 18.420172       |
| 5    | 923     | Citizen Kane             | Mystery Drama                               | 18.305990       |
| 6    | 2858    | American Beauty          | Drama                                       | 18.291699       |
| 7    | 912     | Casablanca               | Drama Romance                               | 18.020795       |
| 8    | 111     | Taxi Driver              | Crime Drama                                 | 17.924403       |
| 9    | 1208    | Apocalypse Now           | Drama War                                   | 17.881439       |
| 10   | 3481    | High Fidelity            | Comedy Drama Romance Music                  | 17.853777       |


| Rank | movieId | Title                   | Genres                               | Rating |
|------|---------|--------------------------|--------------------------------------|--------|
| 1    | 903     | Vertigo                  | Mystery Romance Thriller             | 5.0    |
| 2    | 26151   | Au Hasard Balthazar      | Drama                                | 5.0    |
| 3    | 7064    | Beauty and the Beast     | Drama Fantasy Romance                | 5.0    |
| 4    | 1673    | Boogie Nights            | Drama                                | 4.5    |
| 5    | 1333    | The Birds                | Horror                               | 4.5    |
| 6    | 3307    | City Lights              | Comedy Drama Romance                 | 4.5    |
| 7    | 3160    | Magnolia                 | Drama                                | 4.5    |
| 8    | 1748    | Dark City                | Mystery Science Fiction              | 4.5    |
| 9    | 2692    | Run Lola Run             | Action Drama Thriller                | 4.5    |
| 10   | 1199    | Brazil                   | Comedy Science Fiction               | 4.0    |

#### 🧐 Interpretation – Collaborative Filtering Recommendation (Memory-Based)

Rekomendasi di atas dihasilkan oleh model Collaborative Filtering berbasis memori (item-item), untuk user dengan ID `45`.

##### 🎬 Preferensi User
Berdasarkan histori rating, user 45 menunjukkan ketertarikan kuat terhadap film dengan genre:
- **Drama**, **Mystery**, dan **Thriller**
- Beberapa elemen **Romance**, **Fantasy**, dan **Sci-Fi**
- Menyukai film-film klasik dan sinematik yang kuat secara naratif

Contoh film yang disukai:
- *Vertigo*, *Beauty and the Beast*, *Boogie Nights*, *Magnolia*, *City Lights*

##### 🔍 Insight dari Rekomendasi
Model berhasil memberikan rekomendasi yang sangat **selaras dengan gaya dan selera film user**, seperti:
- *Chinatown*, *L.A. Confidential*, *Citizen Kane* — film klasik dengan genre **Mystery/Drama**
- *Being John Malkovich*, *High Fidelity* — film dengan **narasi unik dan psikologis**
- *Casablanca*, *American Beauty*, *Apocalypse Now* — termasuk film **ikonik dan award-winning** dengan kedalaman karakter

##### 📈 Skor Prediksi
- Skor tertinggi mencapai **19.39**, menandakan model memiliki kepercayaan tinggi terhadap kecocokan film tersebut dengan user.
- Seluruh rekomendasi berada pada rentang skor tinggi, menunjukkan konsistensi model dalam mengenali pola preferensi user dari histori rating-nya.

##### ✅ Kesimpulan
Model Collaborative Filtering berbasis memori berhasil memberikan rekomendasi yang:
- **Personalized dan relevan secara tematik**
- Selaras dengan gaya sinematik user yang cenderung **klasik, emosional, dan naratif**
- Mampu mengenali selera user tanpa perlu mengetahui isi konten film secara eksplisit

Rekomendasi seperti ini sangat cocok untuk user aktif dengan riwayat interaksi yang cukup kaya, dan bisa meningkatkan kepuasan dalam eksplorasi film serupa.



## Kesimpulan
Pada proyek ini, kita telah berhasil membangun dan mengevaluasi dua pendekatan utama dalam sistem rekomendasi film:

---

### 1. **Content-Based Filtering (CBF)**  
CBF menggunakan informasi konten dari film seperti **judul**, **keywords**, dan **genre** untuk menghitung kemiripan antar film. Model ini sangat efektif digunakan saat data interaksi user masih terbatas (cold-start problem).  
Dua varian model yang dibangun:
- **TF-IDF + Cosine Similarity** — menangkap bobot penting kata
- **CountVectorizer + Cosine Similarity** — menghitung frekuensi kata

> 🔎 *Hasil evaluasi menunjukkan bahwa TF-IDF + Cosine Similarity mampu merekomendasikan film yang relevan secara tematik dengan preferensi user.*

---

### 2. **Collaborative Filtering (CF)**  
CF memberikan rekomendasi berdasarkan pola interaksi user dengan film. Pendekatan ini menghasilkan rekomendasi personalized yang lebih tajam seiring bertambahnya data rating.
Model yang dibangun:
- **Memory-Based CF** — menggunakan cosine similarity antar item
- **RecommenderNet** — neural network sederhana berbasis embedding

> *Hasil evaluasi menunjukkan bahwa model Memory-Based memberikan performa RMSE yang lebih baik dibandingkan RecommenderNet dalam kondisi saat ini. Namun, RecommenderNet memiliki potensi untuk dikembangkan lebih lanjut.*

---

**Final Insight**

- **CBF cocok untuk user baru**, karena tidak membutuhkan histori rating.
- **CF cocok untuk user aktif**, karena dapat memberikan rekomendasi yang lebih personal.
- Kombinasi keduanya dapat menjadi fondasi untuk **hybrid recommendation system** di masa depan.

Tahap selanjutnya dari proyek ini dapat mencakup:
- Evaluasi tambahan menggunakan **Precision@K**, **Recall@K**, dan **MAP**
- Eksplorasi **Hybrid Filtering**
- Penerapan sistem dalam bentuk API atau aplikasi interaktif

Sistem rekomendasi yang telah dibangun membuktikan bahwa pendekatan machine learning dapat secara efektif membantu user menemukan film yang sesuai dengan preferensi mereka.





