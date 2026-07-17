from spark_session import get_spark
from data_io import load_data, save_data
from model import train

spark = get_spark()

df = load_data(spark)        # retrieve data
result_df = train(df)        # training model
save_data(result_df, spark)  # save results
