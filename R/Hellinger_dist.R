#' Hellinger distance of two distribution matrices
#' 
#' \code{Hellinger_dist} computes the Hellinger distance of two estimated distribution matrices
#' 
#' @param P one distribution matrix
#' @param Q the other distribution matrix
#' @return the Hellinger distance between P and Q
#' @export
#' 
Hellinger_dist <- function(P, Q){
  
  sum <- 0
  p <- as.vector(P)
  q <- as.vector(Q)
  n <- length(p)
  for (i in 1:n) {
    sum <- sum + (sqrt(p[i])-sqrt(q[i]))^2 + (sqrt(1-p[i])-sqrt(1-q[i]))^2
  }
  
  return(sum/n)
}
