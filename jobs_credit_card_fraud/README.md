# Credit Card Detection Model using NBX-Jobs [Deprecated]

Training an Sklearn model using NBX-Jobs.

* **Data**: [[webpage](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)] [[raw](https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2015-01.csv)] | NYC Yellow cabs dataset for January, 2015
* **Code**: [[Notebook](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)] | A kaggle notebook by [Janio Martinez Bachmann](https://www.linkedin.com/in/janio-martinez-bachmann-26040ba1)

First install `nbox` by running the following command:

```
python3 -m pip install nbox && alias nbx="python3 -m nbox"
```

## Steps

### Step #01

Start by creating a blank job:
```
nbx jobs new jobs_credit_card_fraud
cd jobs_credit_card_fraud
```

### Step #02

Create a new folder which will contain all the source code (code already available):
```
mkdir src && touch src/__init__.py
```

For this example I have implemented code om a [Kaggle Notebook](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets) by [Janio Martinez Bachmann](https://www.linkedin.com/in/janio-martinez-bachmann-26040ba1). It has been broken into multiple partitions:

1. `etl.py`: which will download the dataset, preprocess it based on a couple of strategies and returns the input/target dataframes
2. `models.py`: Can perform a model searching across a couple or algorithms, then do a grid search.
3. `nbx_user.py:CreditCardFraudModelTrainer.upload_model`: will then securely transfer the generated models to a running NBX-Build Instance

### Step 03

Deploy this job and start executing it:

```
python3 exe.py deploy
```
