
library(ggplot2)
library(plotly)
library(plyr)
library(ggpubr)
library(grid)
library(ggthemes)
library(R.matlab)

#+++++++++++++++++++++++++
# Function to calculate the mean and the standard deviation
# for each group
#+++++++++++++++++++++++++
# data : a data frame
# varname : the name of a column containing the variable
#to be summariezed
# groupnames : vector of column names to be used as
# grouping variables
data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = median(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}




#col=c("red", "cyan4", "purple", "yellowgreen")
# col=c("red", "cyan4", "purple",  "green3",  "blue", "brown4")
col=c("red",  "blue", "brown4")
shapes = c(16, 17, 15, 10, 7, 6)


setwd("~/reproduce")

results <- readMat("t10rho8-max5-results.mat")

## distance plot
group <- c('(-inf, -2.5]', '(-2.5, -0.5]', '(-0.5, 0.5]','(0.5, 2.5]', '(2.5, inf)')
group <- rep(group, 3)
group <- factor(group, levels=c('(-inf, -2.5]', '(-2.5, -0.5]', '(-0.5, 0.5]','(0.5, 2.5]', '(2.5, inf)'))


prob.group=c('(0, 0.1]', '(0.1, 0.4]', '(0.4, 0.6]', '(0.6, 0.9]', '(0.9, 1]');
prob.group <- rep(prob.group, 3)
prob.group <- factor(prob.group, levels=c('(0, 0.1]', '(0.1, 0.4]', '(0.4, 0.6]', '(0.6, 0.9]', '(0.9, 1]'))



dist <- c(results$dist.mmgn, results$dist.trace, results$dist.max)
err <- c(results$err.mmgn, results$err.trace, results$err.max)
Method <- rep(c("MMGN",  "TraceNorm", "MaxNorm"), each = length(results$dist.max))
Method <- factor(Method, levels=c("MMGN","TraceNorm", "MaxNorm"))



data <- data.frame(group = group,  # Create example data
                   prob.group = prob.group,
                   dist = dist,
                   err = err,
                   Method = Method)

p1 <- ggplot(data, aes(x=group, y=err, group=Method)) +
  geom_line(aes(color = Method), linewidth=1)+
  geom_point(aes(color = Method, shape = Method), size=2.5)+
  labs(
    y="Relative error", x='Value-Range')+ theme_bw() +
  scale_shape_manual(values=shapes)+
  scale_color_manual(values=col)+
  theme(text = element_text(size = 20))



p2 <- ggplot(data, aes(x=prob.group, y=dist/length(results$m0), group=Method)) +
  geom_line(aes(color = Method), linewidth=1)+
  geom_point(aes(color = Method, shape = Method), size=2.5)+
  labs(
    y="Hellinger distance", x="Probability-Range") + theme_bw()+
  scale_shape_manual(values=shapes)+
  scale_color_manual(values=col)+
  theme(text = element_text(size = 20))


ggarrange(p1, p2, nrow=1, common.legend = TRUE, legend = "top")
