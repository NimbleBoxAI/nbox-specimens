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

from src.etl import ETL
from src.models import GridSearchTrainer

import pickle


class CreditCardFraudModelTrainer(Operator):
  def __init__(self, write_to: str = "build", model_prefix: str = "/", pre_find_best_model: bool = False):
    if write_to not in ["build", "s3"]:
      raise ValueError("write_to must be either 'build' or 's3'")

    super().__init__()
    self.etl = ETL(
      dataset_url = "https://nbox-demo-public.s3.ap-south-1.amazonaws.com/data-samples/creditcard.csv.zip?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEF0aCmFwLXNvdXRoLTEiRzBFAiALErcNRmjP7nu2sLG0YuG9ihENS26VXnbPuUmG6nbe5QIhAKRUbsNIBJp0VCbwvyy3fy%2FKj8%2FBf7cyx%2FcnCw5LnidQKuQCCDYQARoMNTUzOTAwMDQzODA5IgyXAXFrq3c4IVPm04kqwQJWSkFagwjf6EqOJhRYdjiaZvQMLJFpESr0Z2g5gALvn%2BfK5cdJyvGJdsVFj4vwbZHliFKxk9%2FEehqg%2B%2BnJZdL%2BthjnSQGMFlIS5S5eDGBNbvFrD2%2FVLsEw07ZUS2Tj1TxWvwiVUMme%2FB7lIIminbRJdlHDGZspwg636ZNSSGgFwB76dCi93gdiAtBJAm72McwsA4RCgSHdWlufEs5EDKq4TWCmv4Byucn7cxITpqGBqnAJ8emb6obJ1glgVfSNE6FFL9%2Fdc2Ce4m4bw6oamcBlxHMML3rE8F1cg7FexlEvaqR%2BrhSw09N0Xwwq2cP%2B7%2FETxGVqtWcvL3i2xKQzqwS3FhSkCtkDB6lX6wAWMK%2FKRa6jTvdlHHeNkb9ElJINTAHebmlbpIt%2Ff%2BitOt1Zs20VdM%2BoMjcic7KGhqgPW1Nb4VYwiMzwkwY6swKDk3UcfdhZOp7lMDP1R%2BpjVjKm2N9DjDiA0so1rlJH1wBYrEweKtnEWSUYZcsTbmbUuGlhDkUad9gBpJRQflWpSBI6zrShxbgpo5IolfOoEEvUkzAySnjnV3nl%2BGuW%2FTfwXdNj0NUWn6wgbuFPGLLdoWQFv%2B1vrUXnTiPhK7xK1URGuTNk%2Bf%2FMSdly9QMlTmpOODUiY9En01KoDevn%2Br2NJmf0gMvw0VwlGsO4QVy3kAOoJ7VL%2FHwdLr9Hz3lyjz4d0a1wZE5jsbSb1ONXy8BaB%2BYMlG%2F1a%2FlrLfUl8DBUaxkcQmahRp8UIf4KvO%2BF08gfOia38fBFZ32X69n70jLQ9hUgikDPGC%2B6a91Y8xVb22ED3qG0jTuG%2FmdwSRj7y0gFWol68FC6eMwtuTCI0PRkgYZJ&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220511T213329Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAYB5YD5YQ5CNSG6FZ%2F20220511%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Signature=196c9da3c15c592667977b46b4fe833f498c454ff43e5ac03de15cb577b29318",
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