# `data_io.py`  Optimization and Spark  Migration Plan


## `data_io.py` optimization

```python
def save_data(df, mode="append", path=DATA_PATH, allow_overwrite=False):
    valid_modes = ["append", "overwrite", "error", "ignore"]

    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}")

    if mode == "overwrite" and not allow_overwrite:
        raise Exception("Overwrite requires explicit permission")

    df.write.format("delta").mode(mode).save(path)
```

## pyspark Transfer to Apache Spark

### 1. Install Apache Spark

```bash
sudo apt install spark
```

### 2. Add config line in `spark_session.py`

```python
.config("spark.master", os.getenv("SPARK_MASTER", "local[*]"))
```

### 3. Run `train.py` by spark instead of python

```bash
spark-submit train.py
```

## Data Migration

### Install AWS CLI

```bash
sudo apt install awscli
```

### Upload EEG data from local to S3

Notice: substitute the real endpoint and path when using this command.

```bash
aws configure set default.s3.endpoint_url https://s3.us-west-1.idrivee2.com
aws s3 sync /path/to/eeg_data s3://your-bucket/eeg/raw/
```

### Clone data from one S3 bucket to another S3 bucket

Notice: substitute the real path when using this command.

```bash
aws s3 sync s3://source-bucket/eeg/raw/ s3://target-bucket/eeg/raw/
```




