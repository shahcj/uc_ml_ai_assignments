### API Performance Testing Model

**Chintan Shah**

#### Executive summary
As a performance engineer, I want to build a predictive model that estimates system resource usage—such as CPU, memory, and thread utilization—based on the distribution of API usage within an application. Currently, predicting how different API load mixes will impact system resources requires extensive performance testing with multiple load distributions. This project will use Linear Regression (and possibly other regression methods) to model the relationship between API usage patterns and resulting resource consumption. Given input data describing the percentage load distribution across different APIs, the model will predict the expected resource usage. This will help in capacity planning and reduce the need for repeated, costly performance testing scenarios.

#### Rationale
Performance testing is expensive and time-consuming. Capacity planning today relies heavily on manual experiments. Predictive modeling can help Reduce Testing cycles and Support proactive scaling decisions

#### Research Question
Can we accurately predict system resource utilization (CPU, memory, and thread usage) based on the percentage distribution of multiple API load's within an application using regression models?

#### Data Sources
JMeter load test executions Monitoring tool like Argus and Grafana

#### Methodology
Data Preparation: Clean, Normalize, Scaling etc.. 
Model Techniques: Linear regression as base model, Regularization using Lasso or Ridge, Polynomial Regression for non-linear features 
Metrics: Coef, MSE/RMSE, Cross-validation

#### Results
The baseline Linear Regression models performed strongly for all three targets and remained consistent under 5-fold cross-validation. For CPU prediction, the model achieved very low error (test MSE: 0.122; CV RMSE: 0.352 ± 0.003), showing high accuracy and stability. Memory prediction also performed well (test MSE: 26,973.821; CV RMSE: 165.245 ± 2.115), with error magnitude reasonable given the larger MB scale from 200 to 6000 MB. So model is off by about 164 MB on average as shown by RMSE. Thread prediction showed good performance as well (test MSE: 63.886; CV RMSE: 7.971 ± 0.079). Overall, these results indicate that API-level traffic and latency features are effective predictors of system resource usage and provide a solid baseline for further model refinement.

#### Next steps
1. Tune regularized linear models: Ridge/Lasso with hyperparameter search (GridSearchCV) to reduce overfitting and stabilize coefficients.
2. Add more scenarios: low/high traffic phases, burst behavior, deployment windows, failures/retries. This will help to test the model for different edge cases.

#### Outline of project

- [Link to notebook 1]()
- [Link to notebook 2]()
- [Link to notebook 3]()
