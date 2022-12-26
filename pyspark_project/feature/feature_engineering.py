from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline


class FeatureEngineering:
    def __init__(self):
        self.vector_assembler = VectorAssembler(inputCols=['Age',
                                                           'Total_Purchase',
                                                           'Account_Manager',
                                                           'Years',
                                                           'Num_Sites'], outputCol='features')
        self.pipeline = Pipeline(stages=[self.vector_assembler])

    @staticmethod
    def fit_pipeline(pipeline, df_train):
        return pipeline.fit(df_train)

    @staticmethod
    def save_pipeline(pipeline_model, save_path):
        pipeline_model.write().overwrite().save(save_path)

    def fit_and_save_pipeline(self, df_train, save_path):
        pipeline_model = self.fit_pipeline(self.pipeline, df_train)
        self.save_pipeline(pipeline_model, save_path)
