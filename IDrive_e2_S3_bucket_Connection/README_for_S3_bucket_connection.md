# S3 Bucket Integration with Apache Spark (IDrive e2)

> Note: Sensitive credentials are not included in this repository and should be configured locally on the server.

## Original Collaborative Development Repository

```Text
This is a collaborative project. The original development repository is:
https://github.com/7476ZHAO/IDrive_e2_S3_bucket_Connection.git
```

## Overview

This project demonstrates how to integrate Apache Spark with IDrive e2 (S3-compatible object storage) to enable scalable data access and processing via the `s3a://` protocol. It demonstrates an end-to-end data pipeline using Spark and Delta Lake.

---

## Data Loading Flow

```text
Run training.py
    ↓
Initialize Spark (spark_session.py)
    ↓
Call load_data() (data_io.py)
    ↓
Spark reads Delta Lake data from S3 (IDrive e2)
    ↓
❗ Authentication (credentials)
    ↓
Load into DataFrame (df)
    ↓
Pass df to training step (training.py)
```

---

## Data Saving Flow

```text
Training produces result_df
    ↓
Call save_data(result_df) (data_io.py)
    ↓
Spark writes data to S3 (Delta)
    ↓
❗ Authentication (credentials)
    ↓
Data stored in S3
```

---

## Setup

### 0. System Requirements

Ensure the following are installed on your system:

- Python 3.8+
- Java (JDK 8 or 11 or 17)

> PySpark is installed via `requirements.txt` and includes a local Spark runtime.

---

### 1. Configure Server Credentials

Create AWS-style credentials on the server:

```bash
mkdir -p ~/.aws
nano ~/.aws/credentials
```

Add:

```text
[default]
aws_access_key_id=YOUR_ACCESS_KEY
aws_secret_access_key=YOUR_SECRET_KEY
```

Set permissions:

```bash
chmod 600 ~/.aws/credentials
```

---

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```text
S3_ENDPOINT=https://s3.us-west-1.idrivee2.com
DATA_PATH=s3a://your-bucket/jiali/test
```

Create a `config.py` file to load environment variables:

```python
from dotenv import load_dotenv
import os

load_dotenv()

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
DATA_PATH = os.getenv("DATA_PATH")
```

Make sure `.env` is included in `.gitignore` and not committed to the repository.

### 3. Create `spark_session.py` for Spark Configuration

```python
from pyspark.sql import SparkSession
from config import S3_ENDPOINT

def get_spark():
    return (
        SparkSession.builder
        .appName("delta-test")
        .config("spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.3.1,io.delta:delta-spark_2.12:2.4.0")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.endpoint",
                S3_ENDPOINT)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
        .getOrCreate()
    )
```

---

### 4. Create `data_io.py` for Data I/O Functions

```python
from config import DATA_PATH

def load_data(spark):
    return spark.read.format("delta").load(DATA_PATH)

def save_data(df, spark):
    df.write.format("delta").mode("append").save(DATA_PATH)
```

### 5. Install Dependencies (if not already installed)

```bash
pip install -r requirements.txt
```

---

## Testing

This step verifies the end-to-end data pipeline by creating `test_s3_connection.py`, which writes data to and reads data from S3.

```python
from spark_session import get_spark
from data_io import load_data, save_data

spark = get_spark()

# 1. Create a local DataFrame
df = spark.createDataFrame([
    (1, "Alice"),
    (2, "Bob")
], ["id", "name"])

# 2. Write to S3 using function from data_io.py
save_data(df, spark)

# 3. Read back using function from data_io.py
df2 = load_data(spark)

df2.show()
```

---

## Configure S3 Integration for Model Training

This section demonstrates how to integrate S3-based data I/O into the model training workflow.

### 1. Create `model.py` for Model Training

Define your model logic in `model.py`, if you already have one, you can skip this step:

```python
def train(df):
    print("Training started...")
    df.show()
    return df
```

---

### 2. Add Data Loading and Saving to `training.py`

```python
from spark_session import get_spark
from data_io import load_data, save_data
from model import train

spark = get_spark()

# Load data
df = load_data(spark)

# Train model
result_df = train(df)

# Save results
save_data(result_df, spark)
```

---

## Security Design

- Credentials are **not hardcoded** in source code
- Stored securely in `~/.aws/credentials`
- Access is managed via local credential configuration
- Avoids exposing secrets in GitHub repositories

---

## Notes

- Ensure the correct **S3 endpoint** is used (based on the bucket region)
- Bucket paths follow the format:

```text
s3a://bucket-name/folder/path
```

- Delta Lake support is enabled via Spark packages (e.g., delta-spark, hadoop-aws)

---

## Summary

This setup enables:

- Secure access to cloud storage through externalized credentials
- Clean separation between configuration, data I/O, and application logic
- Scalable data processing using Apache Spark
- A modular data pipeline supporting read → process → write workflows

---
