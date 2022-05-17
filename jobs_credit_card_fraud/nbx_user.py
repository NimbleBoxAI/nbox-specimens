# Auto generated user stub using 'nbox jobs new' command
# project name: jobs_taxi_dataset
# created time: Wednesday W19 [ UTC 11 May, 2022 - 09:44:24 ]
#   created by: 
#
# > feeling stuck, start by populating the functions below <

import os
import datetime as dt

os.environ["NBOX_LOG_LEVEL"] = "INFO" # Keep it the way you like

from nbox import Operator
from nbox.lib.aws import S3Operator
from nbox.lib.nbx_instances import NboxInstanceMv
from nbox.hyperloop.job_pb2 import Resource

from src.etl import ETL
from src.models import GridSearchTrainer

import pickle


class CreditCardFraudModelTrainer(Operator):
  def __init__(self, write_to: str = "build", model_prefix: str = "/", pre_find_best_model: bool = False):
    if write_to not in ["build", "s3"]:
      raise ValueError("write_to must be either 'build' or 's3'")

    super().__init__()
    self.etl = ETL(
      dataset_url = "https://nbox-demo-public.s3.ap-south-1.amazonaws.com/data-samples/creditcard.csv.zip",
      sampling_strategy = "random-undersample"
    )
    self.grid_search_trainer = GridSearchTrainer(
      model_to_search_params = {
        "LogisiticRegression": {
          "penalty": ['l1', 'l2'],
          'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        },
        "KNearest": {
          "n_neighbors": list(range(2,5,1)),
          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        "Support": {
          'C': [0.5, 0.7, 0.9, 1],
          'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
        },
        "DecisionTreeClassifier": {
          "criterion": ["gini", "entropy"],
          "max_depth": list(range(2,4,1)),
          "min_samples_leaf": list(range(5,7,1))
        }
      }
    )
    
    # define the uploader
    if write_to == "s3":
      raise NotImplementedError("S3 is not yet implemented")
      self.upload_model = S3Operator(**kwargs,)
    elif write_to == "build":
      self.upload_model = NboxInstanceMv(
        i = os.environ.get("NBOX_INSTANCE_ID"),
        workspace_id = os.environ.get("NBOX_WORKSPACE_ID")
      )

    self.model_prefix = model_prefix
    self.pre_find_best_model = pre_find_best_model

  def forward(self):
    X, y = self.etl()
    model, model_name = self.grid_search_trainer(
      X = X, y = y, find_model = self.pre_find_best_model, model_name = "all",
    )

    # serialise the model
    model_bytes = pickle.dumps(model)
    if not self.model_prefix.endswith("/"):
      self.model_prefix += "/"
    model_path = self.model_prefix + "model.pkl"
    with open(model_path, "wb") as f:
      f.write(model_bytes)

    # upload the model to appropriate location
    self.upload_model(model_path, f"nbx://{model_name}_{dt.datetime.now(dt.timezone.utc).isoformat()}")
    print("Done ...")


def get_op() -> Operator:
  """Function to initialising your job, it might require passing a bunch of arguments.
  Use this function to get the operator by manually defining things here"""
  
  job = CreditCardFraudModelTrainer()
  
  return job

def get_resource() -> Resource:
  """Define your pod config here"""
  return Resource(
    cpu = "100m",         # 100mCPU
    memory = "200Mi",     # MiB
    disk_size = "1Gi",    # GiB
  )
