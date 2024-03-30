
R code for the 1BitTC method in 'A Majorization-Minimization Gauss-Newton Method for 1-Bit Matrix Completion'

This folder includes code for implementing the 1BitTC method (from Wang and Li (2020)) for 1-bit matrix completion in R.


===== Functions =====

Binary_tensor.R implements the 1BitTC method for 1-bit tensor completion, which is available at https://github.com/Miaoyanwang/Binary-Tensor.

Binary_tensor_auto.R implements the 1BitTC method for 1-bit matrix completion by considering matrices as a three-way tensor with the third mode having dimension one. It uses a data-driven approach to for selecting the rank constraint parameter r.


===== References =====

Wang, M. and Li, L. (2020), “Learning from binary multiway data: Probabilistic tensor decomposition and its statistical optimality,” Journal of Machine Learning Research, 21, 6146–6183.



Xiaoqian Liu
xiaoqian.liu1025@gmail.com

Feb. 2024 




  
