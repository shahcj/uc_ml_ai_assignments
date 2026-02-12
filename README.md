**CAPSTONE PROJECT**

As a performance engineer, I want to build a predictive model that estimates system resource usage—such as CPU, memory, and thread utilization—based on the distribution of API usage within an application. Currently, predicting how different API load mixes will impact system resources requires extensive performance testing with multiple load distributions.

This project will use Linear Regression (and possibly other regression methods) to model the relationship between API usage patterns and resulting resource consumption. Given input data describing the percentage load distribution across different APIs, the model will predict the expected resource usage. This will help in capacity planning and reduce the need for repeated, costly performance testing scenarios.*

**PROBLEM STATEMENT AND APPROACH**

Can we accurately predict system resource utilization (CPU, memory, and thread usage) based on the percentage distribution of multiple API load's within an application using regression models?

**Expected Data Source**

JMeter load test executions
Monitoring tool like Argus and Grafana

**Techniques to be used in Analysis**

Data Preparation: Clean, Normalize, Scaling etc..
Model Techniques: Linear regression as base model, Regularization using Lasso or Ridge, Polynomial Regression for non-linear features
Metrics: Coef, MSE/RMSE, Cross-validation

**Expected Results**

The model should identify which APIs have the strongest impact on CPU and memory usage.
Resource usage will likely increase linearly with certain APIs but may show nonlinear effects under high load.
The model will allow estimation of system resource usage for new API load distributions without running new load tests.

**Why this question is important**

Performance testing is expensive and time-consuming. Capacity planning today relies heavily on manual experiments.
Predictive modeling can help Reduce Testing cycles and Support proactive scaling decisions
