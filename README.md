

\# Delhivery Last-Mile Delay Intelligence Pipeline



\## Project Overview



This project analyzes last-mile shipment data to understand delivery delays, predict shipment delivery time, identify shipments at risk of delay, and group hubs based on their operational performance.



The project uses Python, Pandas, NumPy, and Scikit-learn to build a complete data preparation, analysis, machine learning, and prediction pipeline.



\## Project Objectives



\- Clean and prepare shipment data.

\- Analyze delivery delays across service types, hub tiers, and distance ranges.

\- Predict shipment delivery time using regression models.

\- Classify shipments as delayed or not delayed.

\- Identify important features affecting delivery delays.

\- Segment hubs into operational performance groups.

\- Predict delays for new shipments and recommend actions.



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Joblib

\- Git and GitHub



\## Project Structure



```text

Delhivery\_Capstone/

│

├── data/

│   ├── hubs.csv

│   ├── raw\_shipments.csv

│   ├── new\_shipments.csv

│   └── modelling\_table.csv

│

├── src/

│   └── shipment\_prep.py

│

├── models/

│   ├── delay\_model.pkl

│   ├── delay\_threshold.csv

│   ├── feature\_order.csv

│   └── impute\_values.csv

│

├── outputs/

│   ├── part\_a\_checks.csv

│   ├── part\_a\_delay\_by\_group.csv

│   ├── part\_a\_correlations.csv

│   ├── part\_b\_regression\_validation.csv

│   ├── part\_b\_test\_results.csv

│   ├── part\_c\_validation.csv

│   ├── part\_c\_test\_results.csv

│   ├── hub\_profile.csv

│   ├── hub\_segments.csv

│   ├── segment\_profile.csv

│   ├── tier\_vs\_segment\_mismatch.csv

│   └── new\_shipments\_predictions.csv

│

├── generate\_data.py

├── part\_a\_clean\_eda.py

├── part\_b\_regression.py

├── part\_c\_classification.py

├── part\_d\_segments.py

├── package\_model.py

├── predict\_delay.py

├── requirements.txt

└── README.md

```



\## Dataset



The project uses three generated datasets:



\- Hubs: 24 records

\- Raw shipments: 1,224 records

\- New shipments: 48 records



After removing duplicate shipment records, the cleaned dataset contains 1,200 shipments.



The target variable for classification is `is\_delayed`.



A shipment is considered delayed when its actual delivery time exceeds its promised delivery time.



\## How to Run the Project



\### 1. Install dependencies



```bash

py -m pip install -r requirements.txt

```



\### 2. Generate the datasets



```bash

py generate\_data.py

```



\### 3. Run Part A: Data Cleaning and EDA



```bash

py part\_a\_clean\_eda.py

```



\### 4. Run Part B: Delivery Time Regression



```bash

py part\_b\_regression.py

```



\### 5. Run Part C: Delay Classification



```bash

py part\_c\_classification.py

```



\### 6. Run Part D: Hub Segmentation



```bash

py part\_d\_segments.py

```



\### 7. Package the classification model



```bash

py package\_model.py

```



\### 8. Predict delays for new shipments



```bash

py predict\_delay.py

```



The prediction results are saved in:



`outputs/new\_shipments\_predictions.csv`



\## Part A: Data Cleaning and Exploratory Data Analysis



The raw shipment data was cleaned by:



\- Removing duplicate records.

\- Standardizing service types.

\- Handling missing payment modes.

\- Converting declared values into numeric values.

\- Handling invalid parcel weights.

\- Filling missing numerical values using median imputation.



Feature engineering included promised delivery hours, COD indicator, weekend indicator, dispatch hour, and log-transformed declared value.



\### Key Findings



\- Total cleaned shipments: 1,200

\- Delayed shipments: 278

\- Overall delay rate: 23.17%



Express shipments had a delay rate of approximately 48.97%, while Economy shipments had a delay rate of approximately 8.57%.



Tier 3 hubs had a delay rate of 44%, compared with 4% for Tier 1 hubs.



The delay rate increased significantly with shipment distance.



\## Part B: Delivery Time Regression



Regression models were trained to predict actual delivery hours.



Models evaluated:



\- Mean baseline

\- Decision Tree Regressor

\- Linear Regression

\- Random Forest Regressor

\- Gradient Boosting Regressor



The data was divided into training, validation, and testing sets using a 70:15:15 split.



\### Final Test Result



Selected model: Linear Regression



\- MAE: 3.5671 hours

\- MSE: 20.4624

\- RMSE: 4.5235 hours

\- R² Score: 0.9493



The model achieved an R² score of approximately 0.95 on the test dataset.



\## Part C: Delay Classification



Classification models were trained to predict whether a shipment would be delayed.



Models evaluated included:



\- Logistic Regression

\- Decision Tree Classifier

\- Random Forest Classifier

\- Grid Search Random Forest

\- Randomized Search Random Forest



Feature selection identified the following seven important features:



1\. distance\_km

2\. hub\_tier

3\. hub\_daily\_capacity

4\. n\_stops

5\. promised\_hours

6\. rain\_flag

7\. sale\_event\_flag



The selected model was Grid Search Random Forest.



The classification threshold was selected as 0.5 while maintaining validation recall of at least 0.80.



\### Final Test Result



\- Accuracy: 87.22%

\- Precision: 71.43%

\- Recall: 73.17%

\- F1 Score: 72.29%



The model was saved using Joblib and successfully reloaded for verification.



\## Part D: Hub Segmentation



Hubs were grouped into three operational segments:



\- Standard

\- Watch

\- Escalate



The segmentation used hub-level operational measures, including delivery time, delay rate, distance, and number of stops.



\### Segment Summary



| Segment | Hubs | Delay Rate |

|---|---:|---:|

| Standard | 7 | 2.57% |

| Watch | 8 | 15.50% |

| Escalate | 9 | 46.00% |



The analysis also identified hubs whose operational segment differed from their original hub tier.



\## New Shipment Predictions



The saved classification model was used to predict delays for 48 new shipments.



\### Prediction Results



| Prediction | Shipments |

|---|---:|

| Predicted on time | 41 |

| Predicted delayed | 7 |



\### Recommended Actions



| Recommended Action | Shipments |

|---|---:|

| No action | 41 |

| Escalate to hub manager | 5 |

| Proactive customer call | 2 |



These predictions are estimates and should not be treated as confirmed delivery outcomes.



\## Operational Recommendations



\- Monitor Express shipments closely because of their higher observed delay rate.

\- Investigate Tier 3 hubs with consistently high delay rates.

\- Prioritize shipments with long distances and high predicted delay probabilities.

\- Escalate predicted high-risk shipments at Escalate-segment hubs.

\- Contact customers proactively when a shipment is predicted to be delayed at other hubs.

\- Review hub performance regularly using updated shipment data.



\## Production Considerations and Limitations



\- The project uses generated shipment data, not live operational data.

\- Model performance on generated data may not represent real-world performance.

\- Predictions should be monitored against actual delivery outcomes.

\- The classification threshold should be reviewed as business requirements change.

\- A production deployment would require automated data ingestion, monitoring, model retraining, and integration with operational systems.



\## Conclusion



This project demonstrates an end-to-end machine learning pipeline for last-mile delivery analysis.



It combines data cleaning, exploratory analysis, regression, classification, feature selection, hyperparameter tuning, hub segmentation, model packaging, and predictions for new shipments.



The final outputs can help identify potential delivery risks and support more proactive operational decisions.



