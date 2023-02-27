# Using NBX-Projects

NBX-Projects is an MLOps project management tool that helps you see everything about your project in a single page. Using it is super simple, and you can get started in just a few minutes.

This example requires `nbox` install, if you do not have it installed, run the following command:

```bash
pip install nbox
```

This will install the CLI `nbx`.

## 🍇 Step 1: Create a new Project

Go to [NimbleBox dashboard](https://app.nimblebox.ai/) and click on "New Project" button to create your project. Once complete, open it and copy the project ID.

## 🍄 Step 2: Upload sample data

Next we will upload the data to the project, open `upload.py` and update the `project_id` with the one you copied in the previous step. Then run the following command:

```bash
python3 upload.py
```

## 🍉 Step 3: Schedule an experiment

All you have to is upload the code and run the experiment to get required results.

<pre><code>nbx projects --id '< your project id >' - run \
  trainer:train_model \
  <b>--resource_disk_size '10Gi'</b> \
  <b>--n 23</b>
</code></pre>

## 🍍 Step 4: View results

Once the run is complete:
- it will have logged all the results to charts and tables
- all the relevant artifacts will be saved on your cloud buckets
