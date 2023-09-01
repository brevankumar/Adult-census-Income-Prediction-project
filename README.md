## End To End ML Project

### create an environment
```
conda create -p virtual_env_for_adult_census python==3.10

conda activate virtual_env_for_adult_census/
```
### Install all necessary libraries
```
pip install -r requirements.txt
```

### Mongodb Database

```
mongodb+srv://revankumar:revankumar@cluster0.pmcz5li.mongodb.net/
```

### Run data ingestion python file
```
python src/components/data_ingestion.py
```

### Run model trainer python file

```
python src/pipeline/training_pipeline.py
```

### Tracking operations through mlflow

```
mlflow ui
```

### To create a docker image

```
docker build -t testdockerrevan.azurecr.io/adultcensus:latest .

docker login testdockerrevan.azurecr.io

docker push testdockerrevan.azurecr.io/adultcensus:latest
```

### Web app deployed through AWS

```
http://projectdeploy-env.eba-qagymxj6.us-east-1.elasticbeanstalk.com
http://projectdeploy-env.eba-qagymxj6.us-east-1.elasticbeanstalk.com/predict

```
```

```
