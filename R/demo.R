source("MMGN_logist.R")
source("MMGN_probit.R")
source("MMGN_logist_auto.R")
source("MMGN_probit_auto.R")
source("loss_1bit.R")

library(lattice)
library(glmnet)
m <- 1000
n <- 1000
r <- 1

rho <- 0.5
sigma <- 0.5

# generate the underlying matrix, nonspiky
set.seed(2022)
U <- matrix(runif(m*r,min=-0.5,max=0.5),nrow = m)
V <- matrix(runif(n*r,min=-0.5,max=0.5),nrow = n)
M_star <- U%*%t(V)
M_star <- M_star/max(abs(M_star))

sratio <- sqrt(m*n)/norm(M_star, "F")






###############  Logistic model ################
############### ############### ############### 
f <- function(x) {return(plogis(x, scale = sigma))}
gradf <- function(x) {return(dlogis(x, scale = sigma))}


# generate the binary matrix Y
Y0 <- sign(f(M_star) - matrix(runif(m*n), nrow=m))

omega <- sample(1:(m*n), size = floor(rho*m*n))
y <- as.vector(Y0)
y[-omega] <- NA
Y <- matrix(y, m, n)

t <- proc.time()
res_mmgn <- MMGN_logist(Y, omega, R=r, sigma, maxiters=1e3, tol=1e-4, stopping = 'objective')
proc.time()- t


Mhat <- (res_mmgn$U)%*%t(res_mmgn$V)
# relative error
norm(M_star - Mhat, 'F')^2/norm(M_star, 'F')^2


######## *** takes about 1 min ***
t <- proc.time()
res_auto <- MMGN_logist_auto(Y, omega, sigma, rSeq=1:3, rate=0.8, seed=1234, maxiters=5e2, tol=1e-4, stopping = 'objective')
proc.time()- t

Mhat <- (res_auto$U)%*%t(res_auto$V)
# relative error
norm(M_star - Mhat, 'F')^2/norm(M_star, 'F')^2

res_auto$rhat
res_auto$loglik







###############  Probit model ################
############### ############### ############### 
f <- function(x) {return(pnorm(x, sd=sigma))}
gradf <- function(x) {return(dnorm(x, sd=sigma))}


# generate the binary matrix Y
Y0 <- sign(f(M_star) - matrix(runif(m*n), nrow=m))

omega <- sample(1:(m*n), size = floor(rho*m*n))
y <- as.vector(Y0)
y[-omega] <- NA
Y <- matrix(y, m, n)

t <- proc.time()
res_mmgn <- MMGN_probit(Y, omega, R=r, sigma, maxiters=1e3, tol=1e-4, stopping = 'objective')
proc.time()- t


Mhat <- (res_mmgn$U)%*%t(res_mmgn$V)
# relative error
norm(M_star - Mhat, 'F')^2/norm(M_star, 'F')^2


########  *** takes about 1 min ***
t <- proc.time()
res_auto <- MMGN_probit_auto(Y, omega, sigma, rSeq=1:3, rate=0.8, seed=1234, maxiters=5e2, tol=1e-4, stopping = 'estimate')
proc.time()- t

Mhat <- (res_auto$U)%*%t(res_auto$V)
# relative error
norm(M_star - Mhat, 'F')^2/norm(M_star, 'F')^2

res_auto$rhat
res_auto$loglik

