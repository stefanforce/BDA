import math

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


# function for getting the best corelations from the dataset

def get_column_correlations(df_set, correlation_threshold=0):
    columns = [column for column in df_set.schema.names if column not in ["PubMedID", "Rating"]]
    correlations = []

    if correlation_threshold is not None:
        for column in columns:
            correlation = df_set.stat.corr(column, "Rating")
            if math.isnan(correlation) or abs(correlation) < correlation_threshold:
                continue
            correlations.append((column, correlation))

        correlations = list(reversed(sorted(correlations, key=lambda x: abs(x[1]))))

        print("Columns and Correlations", correlations)
        return [cor[0] for cor in correlations]
    return columns


def generate_features(df_set, columns):
    assembler = VectorAssembler(inputCols=columns, outputCol="features")
    df_set = assembler.transform(df_set)
    df_set = df_set.select("features", "Rating")
    return df_set

#generate linear regression model
def generate_model(df_set):
    lr = LinearRegression(featuresCol = 'features', labelCol='Rating', maxIter=10)
    model = lr.fit(df_set)
    return model

def getPredictionsWithGradientBoostedTreeRegression(dataframe_train,dataframe_test):
    from pyspark.ml.regression import GBTRegressor
    gbt = GBTRegressor(featuresCol='features', labelCol='Rating', maxIter=10)
    gbt_model = gbt.fit(dataframe_train)
    gbt_predictions = gbt_model.transform(dataframe_test)
    gbt_predictions.select('prediction', 'Rating', 'features').show(5)

