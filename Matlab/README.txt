Matlab code for 'A Majorization-Minimization Gauss-Newton Method for 1-Bit Matrix Completion'

This folder includes code for implementing the MMGN method for 1-bit matrix completion.



===== Functions =====

MMGN_probit.m implements the MMGN method for 1-bit matrix completion under the probit noise model with a given rank constraint parameter r.
MMGN_logist.m is its counterpart for the logistic noise model.

MMGN_probit_auto.m implements the MMGN method for 1-bit matrix completion under the probit noise model with a data driven approach for selecting 
                   the rank contraint parameter r.
MMGN_logist_auto.m is its counterpart for the logistic noise model.

loss_1bit.m computes the negative log-likelihood / loss for the 1-bit matrix completion.

Hellinger_dist.m computes the Hellinger distance between two distribution matrices P and Q


===== Numerical Experiments =====

demo.m shows two examples to run MMGN under the probit and logistic noise models, respectively.


Xiaoqian Liu
xiaoqian.liu1025@gmail.com

Nov. 2022 
