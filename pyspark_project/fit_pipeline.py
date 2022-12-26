import logging
import logging.config
import os
import argparse

from pyspark.ml import PipelineModel

from loader import DataLoader
from feature import FeatureEngineering, SparkDataFrameConverter
from model import Model


def arg_parser():
    current_working_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='customer churn')
    parser.add_argument('--fe-path', type=str, help='Where to save the feature engineering pipeline',
                        dest='fe_pipeline_save_path',
                        required=False,
                        default=os.path.join(current_working_dir, '../model/fe_pipeline_saprk'))

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
    logger = logging.getLogger(name='customer_churn')

    parse = arg_parser()
    args = parse.parse_args()
    fe_pipeline_save_path = args.fe_pipeline_save_path
    classifier_save_path = args.classifier_save_path
    data_path = args.dataset_path

    logger.info("Load Data")
    df = SparkDataFrameConverter()
    dataset = df.convert(data_path)

    logger.info("Split Data")
    data_set = DataLoader(dataset)
    train_df, test_df = data_set.split_data(train_split=0.8, test_split=0.2)

    logger.info("Fit and Save Feature Engineering Pipeline")
    fe = FeatureEngineering()
    fe.fit_and_save_pipeline(df_train=train_df,
                             save_path=fe_pipeline_save_path)

    logger.info("Load Feature Engineering Pipeline and apply transformations on train set")
    fe_pipeline_model = PipelineModel.load(fe_pipeline_save_path)
    customer_churn = fe_pipeline_model.transform(train_df)

    logger.info("Training a classifier")
    logger.info(customer_churn.select('features','Churn'),)
    model = Model(feature_col="features", label_col="Churn")
    model.fit_and_save_model(df_train=customer_churn,
                             save_path=classifier_save_path)
    logger.info('Model trained and saved')


if __name__ == '__main__':
    main()
