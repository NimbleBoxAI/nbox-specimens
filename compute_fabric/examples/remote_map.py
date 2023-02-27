import os
# os.environ["NBOX_NO_LOAD_GRPC"] = "1"
# os.environ["NBOX_NO_LOAD_WS"] = "1"
# os.environ["NBOX_LOG_LEVEL"] = "warning"

from time import sleep

from nbox import operator, Operator
from nbox.auth import secret

@operator()
def foo(i = 4):
  return i * i

if __name__ == "__main__":
  from fire import Fire

  def main(jid: str, n: int = 5, deploy: bool = False):
    # run locally
    print("-->> foo(10)", foo(10))

    # get the correct operator
    if deploy:
      foo_remote = foo.deploy(group_id = jid) # deploy as batch or API
    else:
      foo_remote = Operator.from_job(job_id = jid)
    
    # map a workload
    _list = list(range(n+1))[1:]
    out = foo_remote.map(_list)
    print(f"-->> foo_remote.map({_list})", out)

  Fire(main)
