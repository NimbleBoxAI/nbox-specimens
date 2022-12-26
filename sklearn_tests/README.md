# 'sklearn_tests' on NimbleBox

In this example we are going to create a batch processing job that will run a benchmark for `sklearn` classification. To deploy this process:

<pre><code>nbx jobs upload main:benchmark --name 'sklearn_test' --trigger

# or you can chose to save the results to relic
nbx jobs upload main:benchmark --name 'sklearn_test' --trigger **--save_to_relic**
</code></pre>
