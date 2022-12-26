# customer_churn_prediction
customer churn prediction using pyspark's mllib

Build Image:

IMAGE_NAME=churn-prediction
IMAGE_VERSION=1.0.0
docker build -t ${IMAGE_NAME}:${IMAGE_VERSION} -f Dockerfile .

To Train a Model:

IMAGE_NAME=churn-prediction
IMAGE_VERSION=1.0.0
PYTHON_SCRIPT=pyspark_project/fit_pipeline.py
docker run -d --name IMAGE_NAME ${IMAGE_NAME}:${IMAGE_VERSION} python ${PYTHON_SCRIPT} ${PARAMS}

To predict:

IMAGE_NAME=churn-prediction
IMAGE_VERSION=1.0.0
PYTHON_SCRIPT=pyspark_project/predict.py
docker run -d --name IMAGE_NAME ${IMAGE_NAME}:${IMAGE_VERSION} python ${PYTHON_SCRIPT} ${PARAMS}


Arguments need to pass:

--fe-path FE_PIPELINE_SAVE_PATH
                        Where to save the feature engineering pipeline
--classifier-path CLASSIFIER_SAVE_PATH
                        Where to save the resulting classifier
----data-path
            where dataset is saved (dataset path)