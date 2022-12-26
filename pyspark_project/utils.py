from pyspark.sql import SparkSession


class SparkSessionHolder:
    def __init__(self):
        pass

    @staticmethod
    def get_spark_session():
        return SparkSession.builder.\
            appName("churn_predictor").\
            getOrCreate()
