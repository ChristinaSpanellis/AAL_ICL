# General Information

All code required to run the project is subdivided into methods.

The best models for feature extraction and feature selection, as well as pre-processed data sets are stored within this repo so the code should run smoothly. 

Before running the code, first install the required dependencies by running `pip install -r requirements.txt`

# How to run the code

- Navigate to the 'running the system' section of the AnomalyDetectionSystem notebook.
- The instantiation of the best models is automatic, running the HH101 section will do the following:
     - preprocess the data
     - perform feature selection
     - perform feature extraction
     - train two different models (one for FS and one for FE) 
     - record MSE of two models
     - inject anomalies
     - make predictions on all of the hold out data for plotting of MSE graphs
     - record anomaly detection metrics for both models (run 10 times and averaged) with varying anomaly score thresholds into the file "hh101_anomalies.txt"
- To test different SEW sizes, the SEW was manually changed for each run.
- NOTE: to enable graph plotting (eg. of MSE per sensor of final prediction MSE) set ENABLE_PLOTS to 1



