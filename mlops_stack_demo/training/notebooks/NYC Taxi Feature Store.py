# Databricks notebook source
# MAGIC %md
# MAGIC # Train an ML model with AutoML and Feature Store feature tables
# MAGIC
# MAGIC In this notebook, you: 
# MAGIC * Create new feature tables in Feature Store 
# MAGIC
# MAGIC ## Requirements
# MAGIC Databricks Runtime for Machine Learning 11.3 or above.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data
# MAGIC
# MAGIC This was generated from the [full NYC Taxi Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

# COMMAND ----------

# Load the `nyc-taxi-tiny` dataset.  
raw_data = spark.read.format("delta").load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")
display(raw_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute features

# COMMAND ----------

from databricks import feature_store
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType, IntegerType, StringType, TimestampType
from pytz import timezone


@udf(returnType=IntegerType())
def is_weekend(dt):
    tz = "America/New_York"
    return int(dt.astimezone(timezone(tz)).weekday() >= 5)  # 5 = Saturday, 6 = Sunday
  
@udf(returnType=StringType())  
def partition_id(dt):
    # datetime -> "YYYY-MM"
    return f"{dt.year:04d}-{dt.month:02d}"


def filter_df_by_ts(df, ts_column, start_date, end_date):
    if ts_column and start_date:
        df = df.filter(col(ts_column) >= start_date)
    if ts_column and end_date:
        df = df.filter(col(ts_column) < end_date)
    return df

# COMMAND ----------

def pickup_features_fn(df, ts_column, start_date, end_date):
    """
    Computes the pickup_features feature group.
    To restrict features to a time range, pass in ts_column, start_date, and/or end_date as kwargs.
    """
    df = filter_df_by_ts(
        df, ts_column, start_date, end_date
    )
    pickupzip_features = (
        df.groupBy(
            "pickup_zip", window("tpep_pickup_datetime", "1 hour", "15 minutes")
        )  # 1 hour window, sliding every 15 minutes
        .agg(
            mean("fare_amount").alias("mean_fare_window_1h_pickup_zip"),
            count("*").alias("count_trips_window_1h_pickup_zip"),
        )
        .select(
            col("pickup_zip"),
            unix_timestamp(col("window.end")).alias("rounded_pickup_datetime").cast(TimestampType()),
            partition_id(to_timestamp(col("window.end"))).alias("yyyy_mm"),
            col("mean_fare_window_1h_pickup_zip").cast(FloatType()),
            col("count_trips_window_1h_pickup_zip").cast(IntegerType()),
        )
    )
    return pickupzip_features
  
def dropoff_features_fn(df, ts_column, start_date, end_date):
    """
    Computes the dropoff_features feature group.
    To restrict features to a time range, pass in ts_column, start_date, and/or end_date as kwargs.
    """
    df = filter_df_by_ts(
        df,  ts_column, start_date, end_date
    )
    dropoffzip_features = (
        df.groupBy("dropoff_zip", window("tpep_dropoff_datetime", "30 minute"))
        .agg(count("*").alias("count_trips_window_30m_dropoff_zip"))
        .select(
            col("dropoff_zip"),
            unix_timestamp(col("window.end")).alias("rounded_dropoff_datetime").cast(TimestampType()),
            partition_id(to_timestamp(col("window.end"))).alias("yyyy_mm"),
            col("count_trips_window_30m_dropoff_zip").cast(IntegerType()),
            is_weekend(col("window.end")).alias("dropoff_is_weekend"),
        )
    )
    return dropoffzip_features  

# COMMAND ----------

from datetime import datetime

pickup_features = pickup_features_fn(
    raw_data, ts_column="tpep_pickup_datetime", start_date=datetime(2016, 1, 1), end_date=datetime(2016, 1, 31)
)
dropoff_features = dropoff_features_fn(
    raw_data, ts_column="tpep_dropoff_datetime", start_date=datetime(2016, 1, 1), end_date=datetime(2016, 1, 31)
)

# COMMAND ----------

display(pickup_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Feature Store library to create new feature tables

# COMMAND ----------

catalog_name = "srijit_nair"
feature_database = "mlops_stack_demo"
spark.sql(f"DROP DATABASE IF EXISTS {catalog_name}.{feature_database} CASCADE");
spark.sql(f"CREATE DATABASE IF NOT EXISTS {catalog_name}.{feature_database}");

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

pickup_features_table = f"{catalog_name}.{feature_database}.trip_pickup_features"
dropoff_features_table = f"{catalog_name}.{feature_database}.trip_dropoff_features"

# COMMAND ----------

#fs.drop_table(pickup_features_table)
#fs.drop_table(dropoff_features_table)

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "5")

fs.create_table(
    name=pickup_features_table,
    primary_keys=["pickup_zip"],
    df=pickup_features,
    timestamp_keys=["rounded_pickup_datetime"],
    description="Taxi Fares. Pickup Features",
)
fs.create_table(
    name=dropoff_features_table,
    primary_keys=["dropoff_zip"],
    df=dropoff_features,
    timestamp_keys=["rounded_dropoff_datetime"],
    description="Taxi Fares. Dropoff Features",
)

# COMMAND ----------


