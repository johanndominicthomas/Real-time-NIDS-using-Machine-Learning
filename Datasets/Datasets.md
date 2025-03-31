Network Intrusion Detection System (NIDS) Project
Overview

This project implements a Network Intrusion Detection System (NIDS) using machine learning algorithms to detect various types of network attacks. The system is trained using the CICIDS 2017 dataset, which contains data about different types of network traffic and attacks.
Dataset

The dataset used in this project is the CICIDS 2017 dataset. It can be downloaded from Kaggle or other similar sources. The dataset consists of multiple CSV files, each representing data from a specific day.
Steps to Prepare the Dataset:

    Download the dataset:
        You can find the CICIDS 2017 dataset on Kaggle(https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset).
        Download all the CSV files corresponding to each day in the dataset.

    Place the files in the Datasets/ directory:
        Create a Datasets/ folder in your project directory.
        Place all the downloaded CSV files into this folder.

    Combine the dataset files:
        Once the dataset files are in the Datasets/ folder, run the following command to combine all the CSV files into a single file:

python combine_datasets.py

This will generate a combined dataset file called combined_cic_ids2017.csv in the project directory.

Create a 10% subset (Optional):

    If you want to use a smaller subset of the combined dataset, you can create a random 10% subset by running the following command:

    python create_subset.py

    This will generate a new file called combined_cic_ids2017_10percent.csv containing 10% of the original dataset.

Requirements

Make sure to install the required dependencies before running the scripts. You can install them using pip with the following command:

pip install -r requirements.txt

This will install all the necessary libraries, including Pandas for data manipulation.
Scripts

    combine_datasets.py: This script combines all the individual CSV files into a single CSV file.
    create_subset.py: This script creates a random 10% subset of the combined dataset.
