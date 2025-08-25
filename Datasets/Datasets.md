# 📂 Network Intrusion Detection System (NIDS) Project - Dataset Setup

## 📖 Overview
This project implements a **Network Intrusion Detection System (NIDS)** using machine learning algorithms to detect various types of network attacks.  
The system is trained using the **CICIDS 2017 dataset**, which contains data about different types of network traffic and attacks.  

---

## 📊 Dataset
The dataset used in this project is the **CICIDS 2017 dataset**.  
It can be downloaded from **Kaggle**.  
The dataset consists of multiple CSV files, each representing data from a specific day.  

---

## 🛠️ Steps to Prepare the Dataset

### 1. Download the dataset:
- You can find the CICIDS 2017 dataset on [Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset).  
- Download all the CSV files corresponding to each day in the dataset.  

---

### 2. Place the files in the `Datasets/` directory:
- Create a folder named `Datasets/` in your project directory.  
- Place all the downloaded CSV files into this folder.  

---

### 3. Combine the dataset files:
- Once the dataset files are in the `Datasets/` folder, run the following command to combine them into a single file:

```
python combine_datasets.py
```

This will generate a combined dataset file called combined_cic_ids2017.csv in the project directory.

---

### 4. Create a 10% subset (Optional)
- If you want to work with a smaller dataset for faster training or testing, you can create a random **10% subset** by running:

```bash
python create_subset.py
```


