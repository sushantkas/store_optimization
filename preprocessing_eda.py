import pandas as pd
class preprocessing_data:
    def __init__(self, feature, store, sales):
        self.feature = feature  # type: pd.DataFrame
        self.store = store  # type: pd.DataFrame
        self.sales = sales
        pass

    def store_preprocessing(self):
        try:
            onehot = pd.get_dummies(self.store["Type"], dtype=int)
            self.store = self.store.join(onehot)  # safer way to add columns
            self.store = self.store.drop("Type", axis=1)
        except Exception as e:
            print(f"Error Occurred During Preprocessing of Store Dataset Column Type: {e}")
        return self.store
    
    def feature_preprocessing(self):
        self.feature['Date'] = pd.to_datetime(self.feature['Date'], format='%d/%m/%Y')
        self.feature = self.feature.sort_values(by=['Store', 'Date'])
        # Preprocessing CPI Column
        if self.feature["CPI"].isna().sum()>0:
            self.feature["CPI"]=self.feature.groupby("Store")["CPI"].apply(lambda x: x.interpolate(method='linear')).values
        # Preprocessing Unemployment Column
        if self.feature["Unemployment"].isna().sum()>0:
            self.feature["Unemployment"] = self.feature.groupby("Store")["Unemployment"].apply(lambda x: x.interpolate(method='linear')).values
        # Preprocessing MarkDown Columns
        if self.feature[["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]].isna().sum().values.sum()>0:
            self.feature[["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]]=self.feature[["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]].fillna(0)
        return self.feature

# need to add function to Join store and feature after preprocessing    