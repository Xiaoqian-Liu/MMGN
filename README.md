
# A Majorization-Minimization Gauss-Newton Algorithm for 1-Bit Matrix Completion

Xiaoqian Liu, Xu Han, Eric Chi, and Boaz Nadler (2023). arxiv: https://arxiv.org/abs/2304.13940. 

This repository provides all the code for running the experiments in the manuscript.  

The Matlab folder includes the Matlab implementation of MMGN, the proposed method for 1-bit matrix completion, and five compared methods, including TraceNorm (Davenport et. al. 2014), MaxNorm (Cai and Zhou, 2013), GD (gradient descent with backtracking), LBFGS, and Manopt (manifold optimization). Please find the detailed description for each method under the eponymous subfolders. 

The R folder includes the R implementation of MMGN for 1-bit matrix completion and two compared methods, including 1BitTC (Wang and Li, 2020) and logisticPCA (De Leeuw, 2006). Please find the detailed description for each method under the eponymous subfolders. 

## References

Cai, T. and Zhou, W.-X. (2013), “A max-norm constrained minimization approach to 1-bit matrix completion,” The Journal of Machine Learning Research, 14, 3619–3647.

Davenport, M. A., Plan, Y., Van Den Berg, E., and Wootters, M. (2014), “1-bit matrix completion,” Information and Inference: A Journal of the IMA, 3, 189–223.

De Leeuw, J. (2006), “Principal component analysis of binary data by iterated singular value decomposition,” Computational Statistics \& Data Analysis, 50, 21–39.

Wang, M. and Li, L. (2020), “Learning from binary multiway data: Probabilistic tensor decomposition and its statistical optimality,” Journal of Machine Learning Research, 21, 6146–6183.