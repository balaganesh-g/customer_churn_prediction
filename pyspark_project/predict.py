import logging
import logging.config
import os
import argparse

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark_project.feature import SparkDataFrameConverter
from pyspark_project.loader import DataLoader


def arg_parser():
    current_working_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Simple project example')
    parser.add_argument('--fe-path',
                        type=str,
                        help="Where to save the feature engineering pipeline",
                        dest='fe_pipeline_save_path',
                        required=False,
                        default=os.path.join(current_working_dir, '../models/fe_pipeline_spark'))
    parser.add_argument('--classifier-path',
                        type=str,
                        help="Where to save the resulting classifier",
                        dest='classifier_save_path',
                        required=False,
                        default=os.path.join(current_working_dir, '../models/classifier_spark'))
    parser.add_argument('--data-path',
                        type=str,
                        help="dataset path",
                        dest='dataset_path',
                        required=True,
                        default=os.path.join(current_working_dir, '../data/customer_churn'))

    return parser


def main():
    logging.config.fileConfig(os.path.abspath('./logging.conf'))
    logger = logging.getLogger(name="customer churn")

    parser = arg_parser()
    args = parser.parse_args()

    fe_pipeline_save_path = args.fe_pipeline_save_path
    classifier_save_path = args.classifier_save_path
    data_path = args.dataset_path

    logger.info("Load Data")
    df = SparkDataFrameConverter()
    dataset = df.convert(data_path)

    logger.info("Split Data")
    data_set = DataLoader(dataset)
    train_df, test_df = data_set.split_data(train_split=0.8, test_split=0.2)

    logger.info("Load Feature Engineering Pipeline and apply transformations on train set")
    fe_pipeline_model = PipelineModel.load(fe_pipeline_save_path)
    test_df = fe_pipeline_model.transform(test_df)

    logger.info("Load classifier and apply predictions")
    nb_model = LogisticRegressionModel.load(classifier_save_path)
    predicted = nb_model.transform(test_df)

    logger.info("Evaluate Results")
    evaluator = BinaryClassificationEvaluator(
        labelCol="label_indexed",
        rawPredictionCol="prediction",
        )

    logger.info("Accuracy on test set : {}".format(evaluator.evaluate(predicted)))


if __name__ == "__main__":
    main()