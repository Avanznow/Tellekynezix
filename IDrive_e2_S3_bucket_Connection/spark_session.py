from pyspark.sql import SparkSession
from config import S3_ENDPOINT

def get_spark():
    return (
        SparkSession.builder
        .appName("delta-test")
        .config("spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.3.1,io.delta:delta-core_2.12:2.4.0")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.endpoint", S3_ENDPOINT)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
        .getOrCreate()
    )


# used for sts
# from pyspark.sql import SparkSession
# import os

# def get_spark():
#     return SparkSession.builder \
#         .appName("delta-test") \
#         .config("spark.jars.packages",
#                 "org.apache.hadoop:hadoop-aws:3.3.1,io.delta:delta-core_2.12:2.4.0") \
#         .config("spark.hadoop.fs.s3a.endpoint",
#                 "https://your-endpoint.idrivee2.com") \
#         .config("spark.hadoop.fs.s3a.path.style.access", "true") \
#         .config("spark.hadoop.fs.s3a.access.key",
#                 os.getenv("AWS_ACCESS_KEY_ID")) \
#         .config("spark.hadoop.fs.s3a.secret.key",
#                 os.getenv("AWS_SECRET_ACCESS_KEY")) \
#         .config("spark.hadoop.fs.s3a.session.token",
#                 os.getenv("AWS_SESSION_TOKEN")) \
#         .getOrCreate()
