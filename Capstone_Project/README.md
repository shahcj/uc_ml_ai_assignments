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
Model Techniques: Linear regression as base model and compare against Random Forest Regressor 
Metrics: Coef, MSE/RMSE, Cross-validation

#### Results
**Linear Regression vs Random Forest Regressor¶**
Random Forest achieved high accuracy on historical data, but it is primarily an interpolation model and does not extrapolate reliably beyond the throughput range seen during training. Since the target scenario involves a total throughput of ~1900, which is outside the observed data, Linear Regression provides a more meaningful estimate of CPU demand for this what-if capacity planning exercise.

#### Conclusion
Using these predictive models, we can estimate CPU requirements under different API throughput combinations without running a new performance test for every scenario. For example, if Graph API throughput increases by 50% while the other APIs stay near their average volume, the model predicts higher CPU demand and indicates that the current CPU core limit may need to be increased. This approach can also be extended to other what-if scenarios, such as peak traffic across all APIs or a significant increase in any single API, to support infrastructure sizing and capacity planning.

#### Outline of project

- [Jupyter Notebook with EDA and first model](https://github.com/shahcj/uc_ml_ai_assignments/blob/main/Capstone_Project/Capstone_Project.ipynb)
