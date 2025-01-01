# Databricks notebook source
# DBTITLE 1,Task 1.1
df = spark.read.csv("/databricks-datasets/nyctaxi/taxizone/taxi_zone_lookup.csv", header=True, inferSchema=True)
df.show(5)


# COMMAND ----------

# DBTITLE 1,Task 1.2
from pyspark.sql.functions import col, sum as spark_sum


null_counts = df.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
null_counts.show()


# COMMAND ----------

# DBTITLE 1,Task 1.3

total_count = df.count()


from pyspark.sql.functions import col, sum as spark_sum
null_counts = df.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
null_counts.show()


null_counts_df = null_counts.toPandas().T  
null_counts_df.columns = ["null_count"]
null_counts_df["null_percentage"] = (null_counts_df["null_count"] / total_count) * 100
print(null_counts_df)


cleaned_df = df.dropna(how="any")


filled_df = df.fillna({
    "Borough": "Unknown",         
    "Zone": "Unknown",            
    "service_zone": "Not Specified"  
})


cleaned_df.show(5)
filled_df.show(5)


# COMMAND ----------

# DBTITLE 1,Task 1.4
from pyspark.sql.functions import col

df.printSchema()

if dict(df.dtypes)["LocationID"] == "string":
    df = df.withColumn("LocationID", col("LocationID").cast("int"))

df.printSchema()


df.show(5)


# COMMAND ----------

# DBTITLE 1,Task 2.1
df.describe().show()


# COMMAND ----------

# DBTITLE 1,Task 2.2
avg_by_category = df.groupBy("Borough").agg({"LocationID": "avg"})  
avg_by_category.show()


avg_by_category_multi = df.groupBy("Borough").agg({"LocationID": "avg", "Zone": "avg"})  
avg_by_category_multi.show()


# COMMAND ----------

# DBTITLE 1,Task 2.3
import matplotlib.pyplot as plt
import seaborn as sns


pandas_df = df.toPandas()


plt.figure(figsize=(10, 6))
sns.boxplot(x=pandas_df["Borough"], y=pandas_df["LocationID"])  
plt.title('Box plot of LocationID by Borough')
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

# DBTITLE 1,Task 2.4
plt.figure(figsize=(10, 6))
sns.countplot(data=pandas_df, x="Borough")  
plt.title('Count of Records by Borough')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(pandas_df["LocationID"], bins=30, kde=True)  
plt.title('Distribution of LocationID')
plt.xlabel('LocationID')
plt.ylabel('Frequency')
plt.show()


# COMMAND ----------

# DBTITLE 1,Task 3.1
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("DataTransformation").getOrCreate()


df = spark.read.csv("/databricks-datasets/nyctaxi/taxizone/taxi_zone_lookup.csv", header=True)


df.show()


# COMMAND ----------

# DBTITLE 1,Task 3.2
from pyspark.sql import functions as F
from pyspark.sql.functions import log, col


df = df.withColumn("log_LocationID", log(col("LocationID") + 1))  


df.select("LocationID", "log_LocationID").show()


# COMMAND ----------

# DBTITLE 1,Task 3.3
df = df.withColumn("date", F.current_date())  


df = df.withColumn("year", F.year(col("date"))) \
       .withColumn("month", F.month(col("date"))) \
       .withColumn("day", F.dayofmonth(col("date")))


df.select("date", "year", "month", "day").show()


# COMMAND ----------

# DBTITLE 1,Task 3.4
df = df.drop("Borough")


df.printSchema()
df.show()


# COMMAND ----------

# DBTITLE 1,Task 4.1
df.createOrReplaceTempView("data_view")

# COMMAND ----------

# DBTITLE 1,Task 4.2 (What is the total count of entries per category?)
# MAGIC %sql
# MAGIC SELECT service_zone AS category, COUNT(*) AS total_count
# MAGIC FROM data_view
# MAGIC GROUP BY service_zone
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Task 4.2 (What is the average value of a specific numeric column, grouped by a categorical column?)
# MAGIC %sql
# MAGIC SELECT service_zone AS category, AVG(log_LocationID) AS average_log_location_id
# MAGIC FROM data_view
# MAGIC GROUP BY service_zone
# MAGIC

# COMMAND ----------

# DBTITLE 1,Task 4.2 (Which categories have values above or below a certain threshold?)
# MAGIC %sql
# MAGIC SELECT service_zone AS category, AVG(log_LocationID) AS average_log_location_id
# MAGIC FROM data_view
# MAGIC GROUP BY service_zone
# MAGIC HAVING AVG(log_LocationID) > 2.0
# MAGIC

# COMMAND ----------

# DBTITLE 1,Task 5.1
import matplotlib.pyplot as plt


pandas_df = df.toPandas()


plt.figure(figsize=(10, 6))
plt.hist(pandas_df['log_LocationID'], bins=30, color='blue', alpha=0.7)
plt.title('Histogram of log_LocationID')
plt.xlabel('log_LocationID')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()


# COMMAND ----------

# DBTITLE 1,Task 5.2
df.createOrReplaceTempView("data_view")

query1 = spark.sql("""
    SELECT service_zone AS category, COUNT(*) AS total_count
    FROM data_view
    GROUP BY service_zone
""")


bar_chart_data = query1.toPandas()


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(bar_chart_data['category'], bar_chart_data['total_count'], color='orange')
plt.title('Count of Entries for Each Service Zone')
plt.xlabel('Service Zone')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# COMMAND ----------

# DBTITLE 1,Task 5.3
trend_data = spark.sql("""
    SELECT date, COUNT(*) AS total_count
    FROM data_view
    GROUP BY date
    ORDER BY date
""")


trend_data_pd = trend_data.toPandas()


plt.figure(figsize=(10, 6))
plt.plot(trend_data_pd['date'], trend_data_pd['total_count'], marker='o', color='green')
plt.title('Trend of Entries Over Time')
plt.xlabel('Date')
plt.ylabel('Total Count')
plt.xticks(rotation=45)
plt.grid()
plt.show()


# COMMAND ----------

# DBTITLE 1,Task 5.4
plt.figure(figsize=(10, 6))
plt.scatter(pandas_df['year'], pandas_df['log_LocationID'], color='red', alpha=0.6)
plt.title('Scatter Plot of log_LocationID vs. Year')
plt.xlabel('Year')
plt.ylabel('log_LocationID')
plt.grid()
plt.show()

