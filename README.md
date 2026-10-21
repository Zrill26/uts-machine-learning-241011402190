# Tugas UTS Machine Learning

Repositori ini berisi pengerjaan Ujian Tengah Semester (UTS) mata kuliah Machine Learning untuk implementasi algoritma Linear Regression dan Logistic Regression secara mandiri menggunakan Python dan NumPy.

---

## Identitas Mahasiswa
- **Nama:** Muhammad Azril Ayatu Syifa
- **NIM:** 241011402190
- **Kelas:** 05TPLM003 Reg B
- **Mata Kuliah:** Machine Learning
- **Random State:** 90 (2 digit terakhir NIM)

---

## Ringkasan Hasil Pengerjaan

### 1. Bagian A — Linear Regression (Dataset: `Ads_Data.csv`)
Memodelkan pengaruh biaya iklan ($X$, dalam juta rupiah) terhadap omset penjualan ($y$, dalam juta rupiah).

- **Persamaan Garis Regresi (Normal Equation):**  
  $$\hat{y} = 27.7371 + 9.0686 \cdot X$$
  - **Intercept ($b = 27.7371$):** Estimasi rata-rata penjualan dasar saat anggaran iklan nol adalah sekitar Rp27,74 juta.
  - **Koefisien ($m = 9.0686$):** Setiap penambahan biaya iklan sebesar Rp1 juta diestimasikan meningkatkan rata-rata penjualan sebesar Rp9,07 juta.

- **Evaluasi Model:**
  | Dataset | MAE (Juta Rp) | MSE | RMSE (Juta Rp) | $R^2$ Score |
  |---|:---:|:---:|:---:|:---:|
  | **Training** | 4.7634 | 40.1032 | 6.3327 | 0.9360 (93.60%) |
  | **Testing** | 4.0134 | 22.2072 | 4.7125 | **0.9644 (96.44%)** |

- **Eksperimen Learning Rate pada Gradient Descent (5000 iterasi):**
  - `lr = 0.001`: Terlalu lambat (underfitting/lambat konvergen, nilai $b=24.8506$, $m=9.5223$).
  - `lr = 0.02`: Paling optimal dan pas, hasil konvergen mendekati Normal Equation ($b=27.7371$, $m=9.0686$).
  - `lr = 0.1`: Divergen / gagal konvergen (terjadi *overflow* karena ukuran langkah terlalu besar sehingga melompati minimum global).

- **Prediksi Kasus Baru:**
  - Biaya Iklan 4.5 Juta $\rightarrow$ Estimasi Penjualan: **Rp68.5458 Juta**
  - Biaya Iklan 7.0 Juta $\rightarrow$ Estimasi Penjualan: **Rp91.2172 Juta**
  - Biaya Iklan 12.0 Juta $\rightarrow$ Estimasi Penjualan: **Rp136.5602 Juta** *(Ekstrapolasi: berada di luar rentang sampel historis sehingga keandalannya berkurang akibat potensi efek kejenuhan pasar)*.

---

### 2. Bagian B — Logistic Regression (Dataset: `Email_Data.csv`)
Klasifikasi email menjadi Spam (1) atau Bukan Spam (0) berdasarkan jumlah link ($X_1$) dan kata promo ($X_2$).

- **Persamaan Probabilitas:**  
  $$P(\text{Spam}) = \sigma(-11.8928 + 1.0922 \cdot \text{JumlahLink} + 0.9559 \cdot \text{KataPromo})$$

- **Persamaan Garis Batas (*Decision Boundary* pada $P=0.5$):**  
  $$\text{KataPromo} = -\frac{-11.8928 + 1.0922 \cdot \text{JumlahLink}}{0.9559}$$

- **Evaluasi Model pada Data Testing:**
  - **Confusion Matrix:**
    $$\begin{bmatrix} 6 & 1 \\ 1 & 7 \end{bmatrix}$$
    *(TN = 6, FP = 1, FN = 1, TP = 7)*
  - **Akurasi:** 0.8667 (86.67%)
  - **Presisi:** 0.8750 (87.50%)
  - **Recall:** 0.8750 (87.50%)
  - **F1-Score:** 0.8750 (87.50%)

- **Eksperimen Ambang Batas (*Threshold*):**
  | Threshold | TN | FP | FN | TP | Precision | Recall | Keterangan |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
  | **0.3** | 5 | 2 | 1 | 7 | 77.78% | 87.50% | Terlalu sensitif (banyak False Positive) |
  | **0.5** | 6 | 1 | 1 | 7 | 87.50% | 87.50% | Default / standar |
  | **0.7 (Dipilih)** | 7 | 0 | 1 | 7 | **100.00%** | 87.50% | **Optimal: FP = 0 (Presisi 100%), mencegah email penting terbuang ke spam** |

- **Prediksi Email Baru (Threshold 0.7):**
  - **E1** (1 link, 2 promo) $\rightarrow P(\text{Spam}) = 0.0001 \rightarrow$ **Bukan Spam (0)**
  - **E2** (5 link, 6 promo) $\rightarrow P(\text{Spam}) = 0.3327 \rightarrow$ **Bukan Spam (0)**
  - **E3** (9 link, 10 promo) $\rightarrow P(\text{Spam}) = 0.9994 \rightarrow$ **Spam (1)**
  - **E4** (2 link, 11 promo) $\rightarrow P(\text{Spam}) = 0.6915 \rightarrow$ **Bukan Spam (0)**

---

### 3. Bagian C — Pemahaman Konsep
- **C1 (Klasifikasi vs Regresi):** Logistic Regression memodelkan regresi linear terhadap *log-odds*, lalu dikonversi menjadi probabilitas kontinu $[0, 1]$ oleh fungsi aktivasi sigmoid. Keputusan kelas diskrit ditentukan lewat nilai ambang batas (*threshold*).
- **C2 (Normal Equation vs Gradient Descent):** Normal Equation menghitung solusi eksak parameter dengan aljabar matriks $\mathcal{O}(d^3)$ yang lambat pada data masif. Gradient Descent memperbarui bobot secara iteratif, sehingga jauh lebih efisien untuk dataset berskala besar.
- **C3 (Data Testing & Overfitting):** Data testing menguji daya generalisasi model pada data yang belum pernah dipelajari. *Overfitting* terjadi saat model menghafal noise data training sehingga akurasi di training tinggi namun anjlok saat diuji di data testing.
- **C4 (MSE vs Log Loss):** Target klasifikasi berbasis probabilitas sigmoid membuat fungsi biaya MSE non-konveks (banyak *local minima*). Log Loss (cross-entropy) digunakan karena menjamin kurva konveks dengan satu *global minimum*.
- **C5 (Perbedaan Hasil Antar Mahasiswa):** Pembagian data train-test bergantung pada `random_state` yang bersumber dari 2 digit terakhir NIM, sehingga komposisi data latih dan parameter bobot yang dihasilkan tiap mahasiswa akan unik.

---

### 4. Bagian Bonus — Perbandingan dengan Scikit-Learn
- **Linear Regression:** Nilai intercept ($27.7371$) dan koefisien ($9.0686$) antara implementasi manual dengan Scikit-Learn bernilai **identik**.
- **Logistic Regression:** Scikit-Learn secara bawaan menerapkan regularisasi L2 (`penalty='l2'`). Saat regularisasi dinonaktifkan (`penalty=None`), bobot Scikit-Learn mendekati hasil manual tanpa penalti bobot.

---

## Struktur Berkas Repositori
```text
uts-machine-learning-241011402190/
├── README.md            # Identitas dan ringkasan hasil evaluasi UTS
├── tugas_UTS.ipynb      # Jupyter Notebook (lengkap dengan output & grafik)
├── Ads_Data.csv         # Dataset Linear Regression
└── Email_Data.csv       # Dataset Logistic Regression
