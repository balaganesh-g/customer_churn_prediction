class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def split_data(self, train_split, test_split):
        df_train, df_test = self.dataset.randomSplit([train_split, test_split])
        return df_train, df_test
