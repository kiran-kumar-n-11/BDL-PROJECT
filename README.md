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
- `Project Report.ipynb`
- `requirements.txt` (to specify modules required and their versions)
- `README.md` (explains the project overview)
- `.git` (lfs storage)


  



