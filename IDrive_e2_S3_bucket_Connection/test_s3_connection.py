from spark_session import get_spark
from data_io import load_data, save_data

spark = get_spark()

# 1 Create a local DataFrame
df = spark.createDataFrame([
    (1, "Alice"),
    (2, "Bob")
], ["id", "name"])

# 2️ Write to S3 using your function
save_data(df, spark)

# 3️ Read back using your function
df2 = load_data(spark)

df2.show()
