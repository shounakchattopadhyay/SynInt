\name{SIMsampler}
\alias{SIMsampler}
\title{Synergistic Interaction Modeling posterior sampler}

\description{
  This function implements the Synergistic Interaction Modeling (SIM) framework in Chattopadhyay et al. (2022+). The dose response surface is decomposed into main effects and pairwise interactions. The pairwise interactions are shrunk towards the space of synergistic or antagonistic interactions. 
}

\usage{
SIMsampler(y, X, Z, K_ME = 5, K_IE = 2, eps_MALA = rep(0.01, choose(dim(X)[2], 2)), c_HMC = 1.0, L_HMC = 5, MC = 10000)
}

\arguments{
  \item{y}{n by 1 vector of continuous responses.}
  \item{X}{n by p matrix of values taken by exposure variables.}
  \item{Z}{n by p_cov matrix of values taken by covariate variables.}
  \item{K_ME}{Number of knots taken to fit main effects. Default value is 5.}
  \item{K_IE}{Number of knots taken to fit interacton effects. Default value is 2.}
  \item{eps_MALA}{p*(p-1)/2 by 1 vector of step-sizes while performing HMC. Default sets all entries to 0.01.}
  \item{c_HMC}{Mass value while performing HMC. Default value is 1.}
  \item{L_HMC}{Number of HMC steps per MCMC iteration. Default is 5.}
  \item{MC}{Number of Markov chain Monte Carlo samples to be drawn from the posterior distribution. Default is 10000.}
  \item{zero_ind}{p*(p-1)/2 by 1 vector with entries = 0 or 1 according to whether the corresponding pairwise interaction is eliminated from the model or included, respectively. Default value rep(1, choose(p,2)) includes all pairwise interactions.}
}

\details{
  A list is returned with the following components.
}

\value{
  \item{intercept}{MC by 1 vector of sampled intercept values.}
  \item{error_var}{MC by 1 vector of sampled error variances.}
  \item{ME_coeff}{MCMC samples of main effect coefficients.}
  \item{ME_scale}{MCMC samples of main effect scales.}
  \item{IE_pos_coeff_part1}{MCMC samples of first part of positive function.}
  \item{IE_pos_coeff_part2}{MCMC samples of second part of positive function.}
  \item{IE_neg_coeff_part1}{MCMC samples of first part of negative function.}
  \item{IE_neg_coeff_part2}{MCMC samples of second part of negative function.}
  \item{ME_mat}{Returns the ME matrix of evaluated spline basis functions.}
  \item{IE_list}{Returns the IE list of spline basis matrices.}
  \item{cov_effects}{MCMC samples of covariate effects.}
  \item{Accept_Prop}{HMC acceptance indicators.}
  \item{HMC_epsilon}{HMC vector of step-sizes after sampling.}
}

\references{
  [to be added]
}

\author{Shounak Chattopadhyay}

\examples{
[to be added]
}
