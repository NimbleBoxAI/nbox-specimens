# Using NBX-Projects

NBX-Projects is a project management tool for managing Machine Learning Operations. With NBX-Projects, you can easily manage your entire project in one place. It's very simple to use, and you can get started in just a few minutes.

To use this example, you need to have nbox installed. If you don't have it already, you can install it by running the following command:

```bash
pip install nbox
```

This will install the CLI `nbx`.

## 🍇 Step 1: Create a new Project

To create a new project, go to the [NimbleBox dashboard](https://app.nimblebox.ai/) and click on the "New Project" button. Once the project is created, open it and copy the project ID.

## 🍄 Step 2: Upload sample data

In the next step, you will upload the data to the project. Run the following command with the `project_id` you copied in the previous step:

```bash
python3 upload.py --project_id '<your project id>'
```

## 🍉 Step 3: Schedule an experiment

To run the experiment and get the desired results, you only need to upload the code and run the experiment. Run the following command in your terminal:

<pre><code>nbx projects --id '< your project id >' - \
  <b>run trainer:train_model</b> \
  --resource_disk_size '10Gi' \
  <b>--n_steps 56</b>
</code></pre>

## 🍍 Step 4: View results

Once the experiment is complete, all the results will be logged in charts and tables, and all the relevant artifacts will be saved on your cloud buckets. You can view the results on the NimbleBox dashboard.

It would look something like:

<img src="https://d2e931syjhr5o9.cloudfront.net/nbox/blackjack-project.png">

## 🍓 Step 5: Deploy the model

To deploy the model as a FastAPI app, run the following command:

<pre><code>nbx serve upload server:app --id 'serving' - \
  <b>serve server:app</b> \
  --resource_disk_size '10Gi' \
  --serving_type "fastapi_v2"
</code></pre>

This will deploy the model on the cloud, and you can start using it right away!
