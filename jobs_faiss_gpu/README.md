# 🔋 'jobs_gpu_faiss' on NimbleBox [Deprecated]

How to run a GPU process using NBX-Jobs.

## What?

[facebookresearch/faiss](https://github.com/facebookresearch/faiss) is a package for similarity search on dense vectors. This requires running a a GPU job using a new `get_resource()` method in `nbx_user.py`.

This job runs all the tests from `faiss` repo sequentially. There are two different strategies you can use:

1. 🏋️‍♀️ `Training` level: check out `nbx_user.py:FaissTests` which directly subclasses `Operator` and runs steps individually
2. 🫑 `Veteran` level: check out `nbx_user.py:FaissTestVeteran` which uses `StepOp` architecture block

Note: that both the strategies will run the same but render different JobsFlow UI.


## How?

```bash
nbx jobs new jobs_gpu_faiss            # create a new job
cd jobs_gpu_faiss
python3 exe.py deploy
```
