import sys
sys.path.append(".")
from pyspark_project.utils import SparkSessionHolder


class SparkDataFrameConverter:
    def __init__(self):
        self.spark = SparkSessionHolder.get_spark_session()

    def convert(self, data_path):
        return self.spark.read.csv(data_path, inferSchema=True, header=True)
