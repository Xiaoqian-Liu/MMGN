R code for the logisticPCA method in 'A Majorization-Minimization Gauss-Newton Method for 1-Bit Matrix Completion'

This folder includes code for implementing the logisticPCA method (from De Leeuw (2006)) for 1-bit matrix completion in R.


===== Functions =====

logistPCA_auto.R implements the logisticPCA method for 1-bit matrix completion under the logistic noise model with a data-driven approach for selecting the rank constraint parameter r.


===== Requirements =====
Our implementation of logisticPCA is built upon the logisticSVD function in the R package logisticPCA, which is available on the CRAN. Please note that logistic PCA only considers the logistic noise model. 



Xiaoqian Liu 
xiaoqian.liu1025@gmail.com

Feb. 2024
