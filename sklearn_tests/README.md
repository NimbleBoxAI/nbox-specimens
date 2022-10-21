# 'sklearn_tests' on NimbleBox

In this example we are going to create a batch processing job that will run a benchmark for `sklearn` classification. To deploy this process:

```bash
nbx jobs upload main:benchmark 'sklearn_test'

# note the ID generated for `sklearn_test` from CLI
nbx jobs --id '<id>' trigger
```

We are continuously working on improving the UX and soon the two commands will be merged into a single more powerful commands.
