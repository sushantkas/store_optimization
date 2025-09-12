import pandas as pd
import pickle
class preprocessing_data:
    def __init__(self, feature, store, sales):
        self.feature = feature  # type: pd.DataFrame
        self.store = store  # type: pd.DataFrame
        self.sales = sales
        try:
            self.immpute=pickle.load(open("impute.pkl","rb"))
        except Exception as e:
            print(f"Error Occurred During Loading Imputer for Nullvalues: {e}")
        pass

    def store_preprocessing(self):
        try:
            onehot = pd.get_dummies(self.store["Type"], dtype=int)
            self.store = self.store.join(onehot)  # safer way to add columns
            self.store = self.store.drop("Type", axis=1)
        except Exception as e:
            print(f"Error Occurred During Preprocessing of Store Dataset Column Name: {e}")
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

    def sales_preprocessing(self):
        self.sales['Date'] = pd.to_datetime(self.sales['Date'], format='%d/%m/%Y')
        self.sales = self.sales.sort_values(by=['Store', 'Date'])
        return self.sales



# need to add function to Join store and feature after preprocessing    


    def merge_dataset(self):
        self.store=self.store_preprocessing()
        self.feature=self.feature_preprocessing()
        self.sales=self.sales_preprocessing()
        self.feature_store=pd.DataFrame()
        self.feature_store=pd.merge(self.store,self.feature,how="left",on="Store")
        self.feature_store_sales= pd.DataFrame()
        self.feature_store_sales=self.feature_store.merge(self.sales,how="left",on=["Store","Date","IsHoliday"])
        self.feature_store_sales['Date_Modified'] = self.feature_store_sales['Date'].astype(int)// 10**9
        self.feature_store_sales["Store"]=self.feature_store_sales["Store"].astype(int)
        return self.feature_store_sales