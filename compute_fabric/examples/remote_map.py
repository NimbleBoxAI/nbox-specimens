import os
# os.environ["NBOX_NO_LOAD_GRPC"] = "1"
# os.environ["NBOX_NO_LOAD_WS"] = "1"
# os.environ["NBOX_LOG_LEVEL"] = "warning"

from nbox import operator
from nbox.auth import secret

@operator()
def foo(i = 4):
  return i * i

if __name__ == "__main__":
  # run locally
  print("-->> foo(10)", foo(10))

  # # run a job
  foo_remote = foo.deploy(os.environ.get("JOB_ID")) # deploy as batch or API
  # out = foo_remote(10)
  # print("-->> foo_remote(10)", out)
  # assert out == 100

  # map a workload
  out = foo_remote.map([1, 2, 3, 4, 5])
  print("-->> foo_remote.map([1, 2, 3, 4, 5])", out)
