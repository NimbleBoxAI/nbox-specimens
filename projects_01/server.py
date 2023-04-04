import os
from typing import List
from fastapi import FastAPI, Response, Request
from pydantic import BaseModel

import torch

from nbox import Project
from nbox.plugins.fastapi import add_live_tracker

from const import project_id
from model import get_model

# download the model if not present and then load and prepare the model
# I don't like this but it works for now
model_fp = "model.pt"
src_fp = "0010-bordeaux-potentiality/model_099/model.pt"
project = Project(project_id)
if not os.path.exists(model_fp):
  project.get_relic().get_from(model_fp, src_fp)
nn = get_model(5, 5)
nn.load_state_dict(torch.load(model_fp))

# create the FastAPI app and add the live tracker
app = FastAPI()
app = add_live_tracker(project, app)

class PredictInput(BaseModel):
  vector: List[float]

@app.post("/predict")
def predict(req: Request, response: Response, data: PredictInput):
  if len(data.vector) != 5:
    response.status_code = 400
    return {"error": "vector should be of length 5"}
  data_tensor = torch.tensor(data.vector).unsqueeze(0)
  with torch.no_grad():
    out = nn(data_tensor)
    pred = out.tolist()
  return {
    'pred': pred
  }

