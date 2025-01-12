# Online Shopper Purchase Prediction

## Summary

1.  Problem Description
2.  Dependency and Environment Management
3.  Exploratory Data Analysis (EDA)
4.  Data Preparation
5.  Model Training and Tuning
6.  Comparing Models' Performance and Selecting the Best
7.  Creating Python Scripts from Notebook
8.  Local Model Deployment with Docker
9.  Cloud Model Deployment with Google Cloud Run

## 1. Problem Description

This project focuses on predicting whether an online shopper will complete a purchase (generate revenue) during a browsing session. This is a binary classification problem where we aim to classify sessions as either leading to a purchase (`True`) or not (`False`). Predicting online shopper behavior is crucial for e-commerce businesses as it allows for targeted marketing, personalized recommendations, and improved user experience, ultimately increasing conversion rates and revenue.

The dataset used for this project is the [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) from the UCI Machine Learning Repository.

## 2. Dependency and Environment Management

This project uses `pipenv` for dependency and environment management. To set up the project environment, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/marvik/ml_zoomcamp_capstone_1.git
    cd ml_zoomcamp_capstone_1
    ```

2.  **Install `pipenv`:**

    ```bash
    pip install pipenv
    ```

3.  **Create a virtual environment with Python 3.12:**

    ```bash
    pipenv --python 3.12
    ```

4.  **Install project dependencies:**

    ```bash
    pipenv install
    ```

5.  **Activate the environment:**

    ```bash
    pipenv shell
    ```

## 3. Exploratory Data Analysis (EDA)

The [Exploratory Data Analysis (EDA)](capstone_1_online_shoppers.ipynb#exploratory-data-analysis-(eda)) section of the notebook provides insights into the dataset. Key findings include:

*   **No Missing Values:** The dataset does not have any missing values.
*   **Target Variable Distribution:** The target variable, `revenue`, is imbalanced, with approximately 84.5% of sessions not resulting in a purchase.
*   **Correlation:**  `pagevalues` shows a strong positive correlation with `revenue`. Other numerical features show varying degrees of correlation with each other.
*   **`specialday`:** Most sessions are not close to a special day.
*   **`visitortype`:** Returning visitors have a higher proportion of sessions resulting in revenue compared to new visitors.

## 4. Data Preparation

In the [Data Preparation](capstone_1_online_shoppers.ipynb#data-preparation) section of the notebook:

*   Column names were converted to lowercase and spaces replaced with underscores for consistency.
*   The data was split into training (60%), validation (20%), and testing (20%) sets.
*   Boolean columns were converted to numerical (0 and 1).
*   Categorical features were one-hot encoded using `DictVectorizer`.

## 5. Model Training and Tuning

Several machine learning models were trained and tuned using the training and validation sets:

*   **Logistic Regression:** Trained with default parameters.
*   **Linear SVM:** Trained with default parameters.
*   **SGDClassifier:** Trained with `log_loss`, L2 penalty, and `alpha=0.001`.
*   **Decision Tree:** Tuned using `max_depth` and `min_samples_leaf`.
*   **Random Forest:** Tuned using `max_depth`, `min_samples_leaf`, and `n_estimators`.
*   **XGBoost:** Tuned using `max_depth`, `eta`, and `min_child_weight`.

## 6. Comparing Models' Performance and Selecting the Best

The models were evaluated based on their AUC scores on the validation set. The **XGBoost** model, after hyperparameter tuning, achieved the highest AUC score and was selected as the best model.

| Model              | AUC      |
| ------------------ | -------- |
| Decision Tree      | 0.9222   |
| Random Forest      | 0.9401   |
| Logistic Regression| 0.9010   |
| Linear SVM         | 0.9108   |
| SGDClassifier      | 0.6283   |
| XGBoost            | 0.9412   |

## 7. Creating Python Scripts from Notebook

The code from the Jupyter Notebook `capstone_1_online_shoppers.ipynb` was used to create two Python scripts:

*   **`train.py`:** Trains the final XGBoost model on the full training data and saves the trained model and `DictVectorizer` to a binary file (`xgboost_model.bin`).

    To run `train.py`:

    ```bash
    pipenv run python train.py
    ```

*   **`predict.py`:** Loads the trained model and `DictVectorizer`, creates a Flask web service to serve predictions via an API endpoint (`/predict`).

## 8. Local Model Deployment with Docker

The prediction service can be deployed locally using Docker:

1.  **Build the Docker image:**

    ```bash
    docker build -t online-shopper-prediction .
    ```

2.  **Run the Docker container:**

    ```bash
    docker run -it -p 9696:9696 online-shopper-prediction
    ```

3.  **Test the service:**
    You can test the service using `predict_sample.py`:

     ```bash
    python predict_sample.py
    ```

## 9. Cloud Model Deployment with Google Cloud Run

The prediction service is deployed to Google Cloud Run using the Docker image stored in Artifact Registry.

**Deployment Steps:**

1.  **Enable Google Cloud services:** Enable the Artifact Registry API and Cloud Build API in your Google Cloud project.
2.  **Create an Artifact Registry repository:** Create a Docker repository in Artifact Registry (e.g., `online-shopper-repo`).
3.  **Build and push the Docker image to Artifact Registry:**

    *   Use Cloud Build and the provided `cloudbuild.yaml` to build the image in the cloud:

        ```bash
        gcloud builds submit --config cloudbuild.yaml .
        ```

4.  **Deploy to Cloud Run:**

    *   **Using the Google Cloud Console:**
        *   Go to Cloud Run and click "Create Service."
        *   Select the image from your Artifact Registry repository.
        *   Configure settings (service name, region, authentication, etc.).
        *   Set the container port to `9696`.
        *   Click "Create."

    *   **Using the `gcloud` command:**
        ```bash
        gcloud run deploy online-shopper-predictor \
          --image=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/online-shopper-repo/online-shopper-prediction:latest \
          --port=9696 \
          --region=us-central1 \
          --allow-unauthenticated
        ```

        *   Replace `YOUR_PROJECT_ID` with your project ID and the image path with your image details.
        

**Service URL:**

The deployed service is accessible at the following URL:

[https://online-shopper-prediction-558797510368.us-central1.run.app/predict](https://online-shopper-prediction-558797510368.us-central1.run.app/predict)

You can find screenshots in screenshots folder

**Testing the Deployed Service:**

1.  Run the script:

    ```bash
    python predict_sample_google_cloud.py
    ```

**Using the Notebook in Google Colab:**

The notebook `capstone_1_online_shoppers.ipynb` was developed and trained in Google Colab. To run the notebook in Colab:

1.  **Mount Google Drive:** You'll need to mount your Google Drive to access the dataset and save the model. The notebook contains the following code snippet to do this:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

    This will prompt you to authorize Colab to access your Google Drive.

2.  **Adjust File Paths:** Update the file paths in the notebook to point to the correct locations within your Google Drive. For example, change the dataset loading path to something like:

    ```python
    file_path = '/content/drive/MyDrive/Colab Notebooks/capstone_1_online_shoppers/online_shoppers_intention.csv'
    ```

    Make sure to adjust other paths (e.g., where you save the model) accordingly.

3.  **Run the Cells:** Execute the cells in the notebook sequentially.

**Files in the Repository:**

*   `capstone_1_online_shoppers.ipynb`: Jupyter Notebook containing the EDA, data preparation, model training, and tuning code.
*   `train.py`: Python script for training the final model.
*   `predict.py`: Python script for creating the prediction service with Flask.
*   `predict_sample.py`: Python script for testing the locally deployed model using Docker.
*   `predict_sample_google_cloud.py`: Python script for testing the deployed model on Google Cloud Run.
*   `xgboost_model.bin`: Saved XGBoost model file and DictVectorizer.
*   `Dockerfile`: Defines the Docker image for the prediction service.
*   `cloudbuild.yaml`: Configuration file for building the Docker image in Google Cloud Build.
*   `Pipfile` and `Pipfile.lock`: Define the project dependencies managed by `pipenv`.
