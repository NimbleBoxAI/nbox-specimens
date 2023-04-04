# 'sklearn_tests' on NimbleBox

In this example, we will demonstrate how to create a batch processing job to run a benchmark for sklearn classification.

- To deploy this process, please follow the steps below:

Run the following command to upload the job:

```bash
nbx jobs upload main:benchmark --name 'sklearn_test' --trigger
```

- Alternatively, if you prefer to save the results to Relic, run the following command:

```bash
nbx jobs upload main:benchmark --name 'sklearn_test' --trigger --save_to_relic
```

Thank you for choosing NimbleBox for your machine learning needs!