from config import DATA_PATH

def load_data(spark):
    return spark.read.format("delta").load(DATA_PATH)

def save_data(df, spark):
    df.write.format("delta").mode("append").save(DATA_PATH)
