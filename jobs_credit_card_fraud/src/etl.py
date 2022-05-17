# ETL is the first step for taxi-dataset

from typing import Tuple
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler # RobustScaler is less prone to outliers.

import nbox
from nbox.lib.shell import ShellCommand
from nbox.lib.comms import Notify
from nbox.lib.exceptions import UserException

class Preprocessing(nbox.Operator):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, df: pd.DataFrame) -> pd.DataFrame:
    """Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)"""
    rob_scaler = RobustScaler()
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
    df.drop(['Time','Amount'], axis=1, inplace=True)

    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']

    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)
    # Amount and Time are Scaled!

    print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
    print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

    return df


class SamplingStrategy(nbox.Operator):
  def __init__(self, strategy = "full", n_splits = 4, random_state = None, shuffle = False) -> None:
    """
    # Splitting the Data (Original DataFrame)
    
    Before proceeding with the Random UnderSampling technique we have to separate the orginal dataframe.
    Why? for testing purposes, remember although we are splitting the data when implementing Random
    UnderSampling or OverSampling techniques, we want to test our models on the original testing set not
    on the testing set created by either of these techniques. The main goal is to fit the model either
    with the dataframes that were undersample and oversample (in order for our models to detect the patterns),
    and test it on the original testing set.

    Args:
      n_splits (int, optional): Number of folds to be used for the StratifiedKFold. Defaults to 4.
      random_state (_type_, optional): Random state to be used for the StratifiedKFold. Defaults to None.
      shuffle (bool, optional): Whether or not to shuffle the data. Defaults to False.
    """
    super().__init__()
    if strategy not in ['full', 'random-undersample']:
      raise UserException("Invalid Strategy")

    self.strategy = strategy
    self.n_splits = n_splits
    self.random_state = random_state
    self.shuffle = shuffle

  def full_split_strategy(self, df: pd.DataFrame):
    sss = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=self.shuffle)

    X = df.drop('Class', axis=1)
    y = df['Class']

    for train_index, test_index in sss.split(X, y):
      print("Train:", train_index, "Test:", test_index)
      original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
      original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    # We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.
    # original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check the Distribution of the labels

    # Turn into an array
    original_Xtrain = original_Xtrain.values
    original_Xtest = original_Xtest.values
    original_ytrain = original_ytrain.values
    original_ytest = original_ytest.values

    # See if both the train and test label distribution are similarly distributed
    _, train_counts_label = np.unique(original_ytrain, return_counts=True)
    _, test_counts_label = np.unique(original_ytest, return_counts=True)

    print('-' * 100)
    print('Label Distributions: \n')
    print(train_counts_label/ len(original_ytrain))
    print(test_counts_label/ len(original_ytest))

    return X, y

  def random_undersample_strategy(self, df: pd.DataFrame):
    df = df.sample(frac=1)

    # amount of fraud classes 492 rows.
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0][:len(fraud_df)]
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # Shuffle dataframe rows
    new_df = normal_distributed_df.sample(frac=1, random_state=42)

    # Undersampling before cross validating (prone to overfit)
    X = new_df.drop('Class', axis=1)
    y = new_df['Class']

    return X, y

  def forward(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if self.strategy == 'full':
      return self.full_split_strategy(df)
    elif self.strategy == 'random-undersample':
      return self.random_undersample_strategy(df)


class ETL(nbox.Operator):
  def __init__(
    self,
    dataset_url: str,
    sampling_strategy: str = "full",
    n_splits: int = 4,
    write_to_file: bool = False,
    target_file_name: str = None,
  ) -> None:
    super().__init__()
    self.dataset_url = dataset_url

    if write_to_file and target_file_name == None:
      raise ValueError("Target File Name is required when writing to file")
    if write_to_file:
      raise NotImplementedError("Writing to file is not yet implemented")
    self.write_to_file = write_to_file
    self.target_file_name = target_file_name

    # generally you should create all the folders that you need here itself
    # this will give you clarity on what are the folders and how that jobs
    # is going to work

    # define the different operations
    self.download = ShellCommand(
      f"wget -q {self.dataset_url} -O raw.csv.zip",
      "unzip raw.csv.zip"
    )
    self.preprocess = Preprocessing()
    self.sampling = SamplingStrategy(
      strategy = sampling_strategy,
      n_splits = n_splits,
    )

  def describe(self, df):
    # The classes are heavily skewed we need to solve this issue later.
    if df.isnull().sum().max() == 0:
      Notify(slack_connect="xxxxx")(
        "The dataframe is not null",
      )
      UserException("There are null values in the dataframe, Exiting!")

    print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
    print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

  def forward(self):
    self.download() # first step is to download the dataset
    df = pd.read_csv("creditcard.csv") # read CSV
    self.describe(df) # print the status

    # preprocess the data
    df = self.preprocess(df)
    self.describe(df) # print the status

    # split the data into X and y
    X, y = self.sampling(df)

    return X, y

