R code for the logisticPCA method in 'A Majorization-Minimization Gauss-Newton Method for 1-Bit Matrix Completion'

This folder includes code for implementing the logisticPCA method (from De Leeuw (2006)) for 1-bit matrix completion in R.


===== Functions =====

logistPCA_auto.R implements the logisticPCA method for 1-bit matrix completion under the logistic noise model with a data-driven approach for selecting the rank constraint parameter r.


===== Requirements =====

Our implementation of logisticPCA is built upon the logisticSVD function in the R package logisticPCA, which is developed by Landgraf, A. J. and Lee, Y. (2020) and available on the CRAN. Please note that logistic PCA only considers the logistic noise model. 


===== References =====

De Leeuw, J. (2006), “Principal component analysis of binary data by iterated singular value decomposition,” Computational Statistics & Data Analysis, 50, 21–39.

Landgraf, A. J. and Lee, Y. (2020), “Dimensionality reduction for binary data through the
projection of natural parameters,” Journal of Multivariate Analysis, 180, 104668.




Xiaoqian Liu 
xiaoqian.liu1025@gmail.com

Feb. 2024
