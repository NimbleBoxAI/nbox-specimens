from nbox import RelicsNBX

import numpy as np
import pandas as pd

if __name__ == "__main__":
  data = np.random.standard_normal((1000, 5))
  labels = np.random.randint(0, 5, 1000)
  pd.DataFrame({
    "a": data[:, 0],
    "b": data[:, 1],
    "c": data[:, 2],
    "d": data[:, 3],
    "e": data[:, 4],
    "label": labels
  }).to_csv("data.csv", index = False)

  RelicsNBX("nbx_core", create = True).put_to(
    "data.csv", f"datasets/data_1000_5.csv"
  )
