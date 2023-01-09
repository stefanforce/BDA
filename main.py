import os
import sys
import warnings

from dataset import transpose_xml_into_dataframe, dataset_csv_to_dataframe
from algo import get_column_correlations
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['HADOOP_HOME'] = "D:\Spark\spark-3.3.1-bin-hadoop3"
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.databricks:spark-xml_2.12:0.13.0 pyspark-shell'


def init_spark():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    spark = SparkSession.builder.master("local").appName("Project_BDA_P_DAT3").getOrCreate()
    sc = spark.sparkContext.setLogLevel("Error")
    return spark, sc


if __name__ == '__main__':
    my_instance_of_spark, sc = init_spark()
    # xml data to dataframe
    process_data = transpose_xml_into_dataframe(my_instance_of_spark, "data.xml", True)

    #print process data headers

    print(process_data.columns)

    # process_data = dataset_csv_to_dataframe(my_instance_of_spark, 'mycsv.csv')

    columns = get_column_correlations(process_data)
    # TBD - ML algorithm
