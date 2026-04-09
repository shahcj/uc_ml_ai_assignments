### API Performance Testing Model

**By Chintan Shah**

#### Executive summary
As a performance engineer, I want to build a predictive model that estimates system resource usage—such as CPU, memory, and thread utilization—based on the distribution of API usage within an application. Currently, predicting how different API load mixes will impact system resources requires extensive performance testing with multiple load distributions. This project will use Linear Regression (and possibly other regression methods) to model the relationship between API usage patterns and resulting resource consumption. Given input data describing the percentage load distribution across different APIs, the model will predict the expected resource usage. This will help in capacity planning and reduce the need for repeated, costly performance testing scenarios.

#### Rationale
Performance testing is expensive and time-consuming. Capacity planning today relies heavily on manual experiments. Predictive modeling can help Reduce Testing cycles and Support proactive scaling decisions

#### Research Question
Can we accurately predict system resource utilization (CPU, memory, and thread usage) based on the percentage distribution of multiple API load's within an application using regression models?

#### Data Sources
JMeter load test JTL file. It spans a 7-day period, where each row represents the average CPU usage and average heap memory percentage recorded at 5-second intervals.

#### Methodology
- Data Preparation: Clean, Normalize, Scaling etc..
- Model Techniques: Linear regression as base model and compare against Random Forest Regressor 
- Metrics: Coef, MSE/RMSE, Cross-validation

#### Results
**Metrics**
- The Linear Regression model showed consistent and reasonably stable performance for CPU prediction. On the test dataset, it achieved an RMSE of **0.648** and an MSE of **0.420**, which means the model’s CPU predictions were off by about **0.65** CPU cores on average. Since the CPU range is 1 to 8, this indicates moderate prediction accuracy and suggests the model can provide useful estimates for capacity planning.
- The 5-fold cross-validation result further supports this, with a mean CV RMSE of **0.645**, which is almost identical to the test RMSE. This close match indicates that the model generalizes consistently across different data splits and shows little evidence of overfitting. Overall, Linear Regression provides a reliable baseline model for estimating CPU demand from throughput.

**Linear Regression vs Random Forest Regressor**
- Random Forest worked very well on the data that was already available, but it is better at predicting values that are similar to what it has already seen. It does not do as well when we give it a much higher throughput value than the ones in the training data. In this case, the target throughput is around **~1,900**, which is outside the range of the existing dataset, so Linear Regression gives a more useful estimate of CPU needed for this future capacity planning scenario.

#### Conclusion
Using these predictive models, we can estimate CPU requirements under different API throughput combinations without running a new performance test for every scenario. For example, if Graph API throughput increases by 50% while the other APIs stay near their average volume, the model predicts higher CPU demand and indicates that the current CPU core limit may need to be increased. This approach can also be extended to other what-if scenarios, such as peak traffic across all APIs or a significant increase in any single API, to support infrastructure sizing and capacity planning.

#### Outline of project

- [Jupyter Notebook with EDA and first model](https://github.com/shahcj/uc_ml_ai_assignments/blob/main/Capstone_Project/Capstone_Project.ipynb)
- [Test Data](https://github.com/shahcj/uc_ml_ai_assignments/blob/main/Capstone_Project/api_perf_metrics.csv)
