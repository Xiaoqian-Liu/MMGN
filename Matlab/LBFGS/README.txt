Matlab code for the LBFGS method in 'A Majorization-Minimization Gauss-Newton Method for 1-Bit Matrix Completion'

This folder includes code for implementing the LBFGS method for 1-bit matrix completion in Matlab.


===== Functions =====

LBFGS_probit.m implements the LBFGS method for 1-bit matrix completion under the probit noise model with a given rank constraint parameter r.
LBFGS_logist.m is its counterpart for the logistic noise model.

LBFGS_probit_auto.m implements the LBFGS method for 1-bit matrix completion under the probit noise model with a data-driven approach for selecting the rank constraint parameter r.
LBFGS_logist_auto.m is its counterpart for the logistic noise model.

func_1bit.m computes the objective and gradient required by the LBFGS method. 



===== Requirements =====
Our implementation of LBFGS is built upon the Matlab toolbox 'poblano', which is available at  https://github.com/sandialabs/poblano_toolbox. Please download the whole package and put the above functions into the main folder to ensure everything works. 



Xiaoqian Liu 
xiaoqian.liu1025@gmail.com

Feb. 2024
