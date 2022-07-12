import os
from nbox import Operator, logger
from nbox.jobs import upload_job_folder

class NowDeploy(Operator):
  def __init__(self, file:str, deployment_id: str, workspace_id: str = None) -> None:
    super().__init__()
    self.folder = os.path.split(os.path.abspath(file))[0]
    self.deployment_id = deployment_id
    self.workspace_id = workspace_id

  def forward(self):
    logger.info(f">>>>>>>>>>> Uploading folder: {self.folder}")
    upload_job_folder(
      method = "serving",
      init_folder = self.folder,
      id_or_name = self.deployment_id,
      workspace_id = self.workspace_id,
    )
    logger.info("If you see this, deployment is complete.")


class BatchPipeline(Operator):
  def __init__(self, file = "./model.pt") -> None:
    super().__init__()
    self.file = file

  def forward(self, text = None) -> None:
    # first step is to create a bunch of files as if these are the weight files for a trained model
    logger.info(f"Writing file: {self.file}")
    with open(self.file, "w") as _f:
      _f.write(text or 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor.')


class ServingPipeline(Operator):
  def __init__(self, file = "./model.pt") -> None:
    super().__init__()
    self.file = file

  def __remote_init__(self):
    with open(self.file, "r") as f:
      self.data = f.read()

  def forward(self, x: str):
    return {"pred": f"{x} :--: {self.data}"}


class MLOps(Operator):
  def __init__(
    self,
    file: str,
    deployment_id: str,
    workspace_id: str = None,
    weights: str = "./model.pt",
  ) -> None:
    super().__init__()
    self.file = file

    self.batch_job = BatchPipeline(weights)
    self.deployer = NowDeploy(__file__, deployment_id, workspace_id)

  def forward(self):
    self.batch_job()
    self.deployer()

