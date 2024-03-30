Matlab code for the NBMF-MM method method in 'A Majorization-Minimization Gauss-Newton Method for 1-Bit Matrix Completion'. The original Python code is available at https://github.com/magronp/NMF-binary.

This folder includes code for implementing the NBMF-MM method proposed by Magron, P. and F ́evotte, C. (2022) for nonnegative binary matrix factorization in Matlab.


===== Functions =====

NBMF.m implements the NBMF-MM algorithm (Algorithm 1 in  Magron, P. and F ́evotte, C. (2022)) for nonnegative binary matrix factorization.

NBMF_auto.m implements the NBMF-MM method for nonnegative binary matrix factorization with a data-driven approach for selecting the rank parameter r.

loss_NBMF.m computes the loss/objective function of the NBMF-MM method. 

perplx_NBMF.m computes the perplexity as defined in Magron, P. and F ́evotte, C. (2022).


===== References =====

Magron, P. and F ́evotte, C. (2022), “A majorization-minimization algorithm for nonnegative binary matrix factorization,” IEEE Signal Processing Letters, 29, 1526–1530.



Xiaoqian Liu 
xiaoqian.liu1025@gmail.com

March. 2024
