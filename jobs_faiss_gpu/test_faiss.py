import os
from nbox import Operator
from nbox.lib.shell import Python


def _get_file(*fp):
  # convert relative fp to absolute fp
  return os.path.join(os.path.dirname(__file__), *fp)


class FaissTests(Operator):
  def __init__(self) -> None:
    super().__init__()

    self.test_1_flat = Python(_get_file('tests', '1-Flat.py'))
    self.test_2_ivfflat = Python(_get_file('tests', '2-IVFFlat.py'))
    self.test_3_ivfpq = Python(_get_file('tests', '3-IVFPQ.py'))
    self.test_4_gpu = Python(_get_file('tests', '4-GPU.py'))
    self.test_5_multiple_gpu = Python(_get_file('tests', '5-Multiple-GPU.py'))

  def forward(self):
    self.test_1_flat()
    self.test_2_ivfflat()
    self.test_3_ivfpq()
    self.test_4_gpu()
    self.test_5_multiple_gpu()
