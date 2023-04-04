import os
from argparse import ArgumentParser
import pandas as pd

from nbox.projects import Project

from model import create_data

def main(pid):
  # Create gaussian data
  data, labels = create_data()

  # Save data to file
  pd.DataFrame({
    "a": data[:, 0].tolist(),
    "b": data[:, 1].tolist(),
    "c": data[:, 2].tolist(),
    "d": data[:, 3].tolist(),
    "e": data[:, 4].tolist(),
    "label": labels.tolist(),
  }).to_csv("data.csv", index = False)

  # Get Relic instance
  p = Project(pid)
  r = p.get_relic()

  # Put file to Relic associated with this project
  r.put_to("data.csv", f"datasets/data_1000_5.csv")

  # Delete file from local machine
  os.remove("data.csv")

if __name__ == "__main__":
  args = ArgumentParser()
  args.add_argument("--project_id", type = str, required = True)
  args = args.parse_args()

  main(args.project_id)
