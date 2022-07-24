import os
import subprocess
import sys

from nbox import Operator, logger
from nbox.lib.shell import ShellCommand

import alphafold.common
import tqdm.notebook
from IPython.utils import io

TQDM_BAR_FORMAT = (
  "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)

SOURCE_URL = (
  "https://storage.googleapis.com/alphafold/alphafold_params_colab_2022-01-19.tar",
)
PARAMS_DIR = "alphafold/data/params"
PARAMS_PATH = os.path.join(PARAMS_DIR, os.path.basename(SOURCE_URL))
ALPHAFOLD_COMMON_DIR = os.path.dirname(alphafold.common.__file__)

class Downloader(Operator):
  def __init__(self):
    super().__init__()
    # Download and store stereo_chemical_props.txt
    self.get_chemicals_txt = ShellCommand('''
      mkdir -p ~/content/alphafold/alphafold/common ;
      mkdir -p /opt/conda/lib/python3.7/site-packages/alphafold/common/ ;
      wget -q -P ~/content/alphafold/alphafold/common https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt ;
    ''')

    # Download alphafold_params_colab_2021-10-27.tar
    self.downlaod_params = ShellCommand(f'''
      cp -f ~/content/alphafold/alphafold/common/stereo_chemical_props.txt "{ALPHAFOLD_COMMON_DIR}" ;
      mkdir --parents "{PARAMS_DIR}" ;
      wget -O "{PARAMS_PATH}" "{SOURCE_URL}" ;
    ''')

    # Un-tar alphafold_params_colab_2021-10-27.tar
    self.untar_params = ShellCommand(f'''
      tar --extract --verbose --file="{PARAMS_PATH}" --directory="{PARAMS_DIR}" --preserve-permissions ;
    ''')


  def forward(self):
    with tqdm.tqdm(total=100, bar_format=TQDM_BAR_FORMAT) as pbar:
      self.get_chemicals_txt()
      pbar.update(18)
      self.downlaod_params
      pbar.update(27)
      self.untar_params
      pbar.update(55)
