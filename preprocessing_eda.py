import pandas as pd
import pickle
from sklearn.impute import KNNImputer
class preprocessing_data:
    def __init__(self, feature, store, sales):
        self.feature = feature  # type: pd.DataFrame
        self.store = store  # type: pd.DataFrame
        self.sales = sales
        self._int_columns =["Store","Size","A","B","C","IsHoliday","Dept"]
        pass

    def store_preprocessing(self):
        """
        Preprocess the store dataset

        This function takes the store dataset and preprocesses the Type
        column by one-hot encoding.

        Returns:
            pd.DataFrame: The preprocessed store dataset

        Raises:
            Exception: If the Type column is not found in the store
                dataset, an exception is raised with an error message.
        """
        try:
            onehot = pd.get_dummies(self.store["Type"], dtype=int)
            self.store = self.store.join(onehot)  # safer way to add columns
            self.store = self.store.drop("Type", axis=1)
        except Exception as e:
            print(f"Error Occurred During Preprocessing of Store Dataset Column Name: {e} not found in Store Dataset")
        return self.store
    
    def feature_preprocessing(self):
        """
        Preprocess the feature dataset

        This function takes the feature dataset and converts the Date column
        to datetime format. It also sorts the dataset by the Store and Date
        columns.

        Additionally, it preprocesses the CPI, Unemployment, and MarkDown
        columns by interpolating missing values with a linear function.

        Returns:
            pd.DataFrame: The preprocessed feature dataset
        """
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
        """
        Preprocess the Date column of the sales dataset

        This function takes the sales dataset and converts the Date column
        to datetime format. It also sorts the dataset by the Store and Date
        columns.

        Returns:
            pd.DataFrame: The preprocessed sales dataset
        """
        # Convert the Date column to datetime format
        self.sales['Date'] = pd.to_datetime(self.sales['Date'], format='%d/%m/%Y')
        # Sort the dataset by the Store and Date columns
        self.sales = self.sales.sort_values(by=['Store', 'Date'])
        return self.sales

    def date_column(self):
        """
        Preprocess the Date column of the sales dataset

        This function takes the sales dataset and converts the Date column
        to datetime format. It also sorts the dataset by the Store and Date
        columns.

        Returns:
            pd.DataFrame: The preprocessed sales dataset
        """
        self.sales['Date'] = pd.to_datetime(self.sales['Date'], format='%d/%m/%Y')
        self.sales = self.sales.sort_values(by=['Store', 'Date'])
        return self.sales


    def merge_dataset(self, fillna=False):
        """ After merging Datasets If want to Fill the null values of Dataset with KNN Imputer then set fillna=True """
        self.store=self.store_preprocessing()
        self.feature=self.feature_preprocessing()
        self.sales=self.sales_preprocessing()
        self.feature_store=pd.DataFrame()
        self.feature_store=pd.merge(self.store,self.feature,how="left",on="Store")
        self.feature_store_sales= pd.DataFrame()
        self.feature_store_sales=self.feature_store.merge(self.sales,how="left",on=["Store","Date","IsHoliday"])
        self.feature_store_sales['Date_Modified'] = self.feature_store_sales['Date'].astype(int)// 10**9
        #self.feature_store_sales["Store"]=self.feature_store_sales["Store"].astype(int)

        if fillna ==True:
            self._imputer=KNNImputer(n_neighbors=5)
            self._New_Data=self._imputer.fit_transform(self.feature_store_sales.drop("Date", axis=1))
            self.feature_store_sales[self.feature_store_sales.drop("Date", axis=1).columns]=pd.DataFrame(self._New_Data,columns=self.feature_store_sales.drop("Date", axis=1).columns)
            self.feature_store_sales[self._int_columns]=self.feature_store_sales[self._int_columns].astype(int)
        return self.feature_store_sales.drop("Date_Modified", axis=1, inplace=True)
    

    
    def dates_features(self):
        """
        Create new columns for the date features of the feature_store_sales dataset

        This function takes the feature_store_sales dataset and creates new columns for
        the date features such as year, month, day, week, and day of week.

        Returns:
            pd.DataFrame: The feature_store_sales dataset with the new date feature columns
        """
        d={ 
        "Year":self.feature_store_sales["Date"].dt.year, 
        "Month":self.feature_store_sales["Date"].dt.month, 
        "Day":self.feature_store_sales["Date"].dt.day, 
        "Week":self.feature_store_sales["Date"].dt.isocalendar().week, 
        "DayofWeek":self.feature_store_sales["Date"].dt.dayofweek }
        self.feature_store_sales[['Year', 'Month', 'Day', 'Week', 'DayofWeek']]=pd.DataFrame(data=d, index=self.feature_store_sales.index)
        return self.feature_store_sales