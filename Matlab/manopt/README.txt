Matlab code for the Manopt method in 'A Majorization-Minimization Gauss-Newton Method for 1-Bit Matrix Completion'

This folder includes code for implementing the manifold optimization (Manopt) method for 1-bit matrix completion in Matlab.


===== Functions =====

Manopt_probit.m implements the Manopt method for 1-bit matrix completion under the probit noise model with a given rank constraint parameter r.
Manopt_logist.m is its counterpart for the logistic noise model.

Manopt_probit_auto.m implements the Manopt method for 1-bit matrix completion under the probit noise model with a data-driven approach for selecting the rank constraint parameter r.
Manopt_logist_auto.m is its counterpart for the logistic noise model.


===== Requirements =====
Our implementation of Manopt is built upon the Matlab toolbox 'manopt', which is available at  https://github.com/NicolasBoumal/manopt. Please download the whole package and put the above functions into the 'examples' subfolder to ensure everything works. You also need to execute 'importmanopt.m' before running the code. Please refer to the README.text in the manopt package. 




Xiaoqian Liu 
xiaoqian.liu1025@gmail.com

Feb. 2024
