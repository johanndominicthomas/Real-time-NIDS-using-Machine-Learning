# Real-Time Network Intrusion Detection System Using Machine Learning

-  This project uses a self trained **XgBoost Classifier** model to detect anomalies in real time network traffic.
-  The model was trained using the **CICIDS2017 Dataset**.
-  **Scapy** is used to capture real time packets which is aggregated into a flow which is passed to the XgBoost Model.
-  If anomalies are detected, then they are displayed on the **dashboard**.

---

## 🚀 Setup Instructions

###1. Clone the git repository:
-  If 'git' and 'pip' are not installed ,please install them first.
-  Open a terminal or command prompt and run:
```
git clone https://github.com/johanndominicthomas/Real-time-NIDS-using-Machine-Learning.git
cd Real-time-NIDS-using-Machine-Learning/
```

###2. Installing the requirements:
-  Create a virtual environment:
```
python -m venv venv
```

-  Activate the virtual environment:
**On Windows:**
```
venv\Scripts\activate

```
**On Linux:**
```
source venv/bin/activate

```

Installing the requirements:
```
pip install -r requirements.txt
```



###3. To set up the Datasets, go to [Datasets.md](https://github.com/johanndominicthomas/Real-time-NIDS-using-Machine-Learning/blob/master/Datasets/Datasets.md)


###4. Preprocess the dataset:
```
python preprocessing_classifier.py
```

###5. Create the attack simulation csv:
```
python preprocessing_attack.py
```

###6. Now to run the project:
**On Linux:**
```
sudo venv/bin/activate main.py
```

**On Windows:**

Scapy requires the **Npcap driver** to access raw sockets.  

#### Install Npcap:
1. Go to the [Npcap download page](https://npcap.com/#download).  
2. Download the latest installer.  
3. Run the installer and make sure to check the following boxes:  
   - ✅ Install Npcap in WinPcap API-compatible mode  
   - ✅ Allow non-admin users to capture packets  

Now to run the project, run the follwing command:
```
venv\Scripts\python main.py

```

-  Click on the url to go to the Dashboard.
