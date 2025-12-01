 Battery Health Monitoring & RUL Prediction — Work in Progress 

This project focuses on predicting State of Health (SOH) and Remaining Useful Life (RUL) of Lithium-ion batteries using Machine Learning techniques. The dataset is inspired by the NASA Ames Prognostics Center of Excellence lithium-ion battery degradation experiments.

Dataset Overview
We are using three virtual battery cells:
B0005 (B5)
B0006 (B6)
B0007 (B7)

The dataset (battery_dataset.csv) includes cycle-wise averaged battery characteristics:
Feature	Description
Charging/Discharging Current	Current variation across cycles
Charging/Discharging Voltage	Voltage decline with battery aging
Charging/Discharging Temperature	Thermal behavior during charge-discharge
BCt - Battery Capacity	Key indicator of degradation
SOH - State of Health (%)	Normalized healthy condition
RUL - Remaining Useful Life	Estimated cycles left before failure
These features simulate battery aging patterns and degradation over time.

Project Goal
To build a predictive maintenance model capable of:
- Estimating Battery SOH
- Predicting the RUL with high accuracy
- Supporting EV/Car Battery Health Monitoring systems

Future extensions include:
Neural networks for degradation modeling
Real-time monitoring dashboard
Deployment for smart vehicle applications
