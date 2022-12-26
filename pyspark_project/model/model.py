from pyspark.ml.classification import LogisticRegression


class Model:
    def __init__(self, feature_col='features', label_col='Churn'):
        self.features_col = feature_col
        self.label_col = label_col
        self.clf = LogisticRegression(featuresCol=self.features_col, labelCol=self.label_col)

    @staticmethod
    def fit_model(clf, df_train):
        return clf.fit(df_train)

    @staticmethod
    def save_model(clf_model, save_path):
        clf_model.write().overwrite().save(save_path)

    def fit_and_save_model(self, df_train, save_path):
        clf_model = self.fit_model(self.clf, df_train)
        self.save_model(clf_model, save_path)


