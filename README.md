# Lasso_Regression - Ad Sales Prediction

## Overview

This project implements a **Lasso Regression Model** to predict product sales based on advertising budgets spent on TV, Radio, and Newspaper channels.

The model is trained using a custom dataset and deployed through a **Flask** web application, allowing users to input advertising budgets and get instant predictions.

## Project Structure

```
DataScience/
│
├── Ridge and Lasso/
│   ├── model/
│   │   └── lasso_regression_model.pkl
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   ├── advertising_dataset.csv
│   ├── lasso_model.py
│   ├── app.py
│   └── requirements.txt
```

## Installation & Setup

1.  **Clone the repository**

    ```
    git clone <your-repo-url>
    cd "DataScience/Ridge and Lasso"
    ```

2.  **Create a virtual environment (recommended)**

    ```
    python -m venv venv
    source venv/bin/activate    # For Linux/Mac
    venv\Scripts\activate      # For Windows
    ```

3.  **Install dependencies**

    ```
    pip install -r requirements.txt
    ```

## Dataset

The dataset contains advertising budgets and corresponding sales with the following features:

* **TV** (numeric): Advertising budget for TV in thousands of dollars ($k).
* **Radio** (numeric): Advertising budget for Radio in thousands of dollars ($k).
* **Newspaper** (numeric): Advertising budget for Newspaper in thousands of dollars ($k).
* **Sales** (Target): The sales in thousands of units ($k).

## Problem Statement

Accurately predicting sales based on advertising spend is a crucial business task. This project aims to use a machine learning model to automate this process, providing a quick and reliable way to estimate the potential impact of different ad campaigns.

## Why Lasso Regression?

* **Feature Selection:** Lasso regression is a type of linear regression that performs both variable selection and regularization to enhance the prediction accuracy and interpretability of the statistical model. It's especially useful when you have many features and suspect that only a few of them are actually important.
* **Sparsity:** It shrinks the coefficients of less important features to exactly zero, effectively removing them from the model. This makes the model simpler and easier to interpret.
* **Overfitting Prevention:** By penalizing large coefficients, Lasso helps to prevent the model from overfitting to the training data.

## How to Run

1.  **Train the Model**

    ```
    python lasso_model.py
    ```

    This will create:

    * `lasso_regression_model.pkl` (trained model)

2.  **Run the Flask App**

    ```
    python app.py
    ```

    Visit `http://127.0.0.1:5000/` in your browser.

## Frontend Input Example

Example advertising budget input:

```
TV Ad Budget ($k): 250.45
Radio Ad Budget ($k): 35.21
Newspaper Ad Budget ($k): 10.87
```

## Prediction Goal

The application predicts the sales value in thousands of units, for example: `15.4`.

## Tech Stack

* **Python** – Core programming language
* **Pandas & NumPy** – Data manipulation
* **Scikit-learn** – Machine learning model training
* **Flask** – Web framework for deployment
* **HTML/CSS** – Frontend UI design

## Future Scope

* Deploy the model on a cloud platform like Heroku or Render for public access.
* Add a visualization to the result page, such as a bar chart showing the relative importance of each advertising channel's coefficient.
* Add a feature to allow the user to adjust the `alpha` parameter (the regularization strength) to see how it affects the predictions.


## Screen Shots

**Home Page:**

<img width="1920" height="1080" alt="Screenshot (35)" src="https://github.com/user-attachments/assets/6ab85409-e57b-4ebb-bd56-13c5112486b7" />



**Result Page:**

<img width="1920" height="1080" alt="Screenshot (36)" src="https://github.com/user-attachments/assets/ac4fcea2-2286-47f7-9c7e-f3348d141b00" />
