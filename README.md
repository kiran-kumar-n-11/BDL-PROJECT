# BDL-PROJECT

The project involves using Obesity Levels Dataset to predict the obesity level given few features. MLFlow is used to track the experiments of the several models tried, to find the best model to use it for inference. The models used are XGBoost, Extra Trees classifier, Decision Tree Classifier and Random Forest Classifier. Git-LFS is used for source control. Prometheus and Grafana are used for monitoring. The application is dockerized.

# Project Structure

- `mlartifacts` (contains the artifacts created)
- `mlruns` (contains different runs used for tracking using mlflow)
- `prometheus`
    - `prometheus.yml` (yml file for service
- `Dockerfile` (to build fastAPI image)
- `docker-compose.yaml` (to specify services for multi-container runs)
- `Obesity Levels.csv` (dataset)
- `ObesityLevels.ipynb` (jupy notebook trying out different models)
- `Project Report.ipynb` (Contains mlflow metrics comparision)
- `requirements.txt` (to specify modules required and their versions)
- `README.md` (explains the project overview)
- `.git` (lfs storage)
- `xgboost_model.pkl` (Saved model)
- `app.py` (FastAPI code with prometheus integration)
- `SavingXGBoost.ipynb` (To save xgboost model)
  
# How to use?
- Clone the repo to your local machine.
- Open the terminal and go to the folder where you cloned.
- Use cmd `docker-compose up --build`.
- Then go to the urls
      - For fastapi: `localhost:8000`
      - For prometheus: `localhost:9090`
      - For grafana: `localhost:3000`
- To use SWAGGER UI, go to `localhost:8000/docs` and select POST and then use `try it out` option.
- Once everything is done, you can use `docker-compose down` to close the containers.



