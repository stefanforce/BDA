import os
import sys
import warnings

from dataset import transpose_xml_into_dataframe, dataset_csv_to_dataframe
from algo import get_column_correlations, generate_features, generate_model
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
    # process_data = transpose_xml_into_dataframe(my_instance_of_spark, "data.xml", True)


    process_data = dataset_csv_to_dataframe(my_instance_of_spark, 'mycsv.csv')

    columns = get_column_correlations(process_data)

    features = generate_features(process_data, columns)

    splits = features.randomSplit([0.8, 0.2])  # 80% 20% is the recomended way to split data for training machine
    train_df = splits[0]
    test_df = splits[1]

    lr_model = generate_model(train_df)

    print("Coefficients: " + str(lr_model.coefficients))

    trainResult = lr_model.summary

    print("numIterations: %d" % trainResult.totalIterations)
    print("objectiveHistory: %s" % str(trainResult.objectiveHistory))
    trainResult.residuals.show()

    # RMSE - root mean square error - indicator folosit pentru a analiza diferenta intre valorile preziste de model si valorile reale
    print("rmse - %f" % trainResult.rootMeanSquaredError)
    print("r-square: %f" % trainResult.r2)

    features.describe().show()

    predictions = lr_model.transform(test_df)

    # Predictiile cu obtinute cu linear regression, in csv
    predictions.select("prediction", "Rating", "features").toPandas().to_csv("test.csv")

    predictions.select("prediction", "Rating", "features").show(10)

    # predicted_score.show()


    # TBD - ML algorithm
