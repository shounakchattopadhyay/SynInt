#### Load Packages ####

# library(bayesplot)
# library(MASS)
# library(splines)
# library(gam)
# library(splines2)
# library(truncnorm)
# library(mvtnorm)
# library(Rcpp)
# library(RcppArmadillo)

#source("penmatt.R")
#sourceCpp("cpp_MCMCSampler_adaptive_optimized.cpp")

# ### Univariate Penalty
# 
# penmatt<-function(M)
# {
# 
#   Amat = matrix(0, nrow = M, ncol = M)
# 
#   Amat[1,] = c(1, rep(0, M-1))
# 
#   for(i in 2:M)
#   {
# 
#     Amat[i,] = c(rep(0, i-2), -1, 1, rep(0, M-i))
# 
#   }
# 
#   return(t(Amat) %*% Amat)
# 
# }

### Bivariate Penalty

penmatt<-function(M)
{

  Amat = matrix(0, nrow = M, ncol = M)

  Amat[1,] = c(1, rep(0, M-1))
  Amat[2,] = c(-2, 1, rep(0, M-2))

  for(i in 3:M)
  {

    Amat[i,] = c(rep(0, i-3), 1, -2, 1, rep(0, M-i))

  }

  return(t(Amat) %*% Amat)

}

##M+4 = #splines for main effects, N+3 = #splines for interaction tensor products

SIMsampler<-function(y,
                     X, 
                     Z = NULL, 
                     K_ME = 5,
                     K_IE = 2, 
                     a_lamb = 0.001,
                     b_lamb = 0.001,
                     eps_MALA = rep(0.01, choose(dim(X)[2], 2)),
                     c_HMC = 1, 
                     L_HMC = 5, 
                     MC = 10000,
                     zero_ind = rep(1, choose(dim(X)[2], 2)),
                     me_integral_constraint = TRUE,
                     cutoff = 0.5*MC,
                     accept_low = 0.65,
                     accept_high = 0.9,
                     accept_scale = 0.8){
  
  library(MASS)
  library(splines)
  library(splines2)
  library(mvtnorm)
  
  #### Samples from the model
  #### E(y_i | X_{1:p}) = \alpha +
  #### \sum_{j=1}^{p} f_j(X_{j,i}) +
  #### \sum_{u < v} h_{uv}(X_{u, i}, X_{v, i}) +
  #### \epsilon_i.
  
  n = dim(X)[1]
  p = dim(X)[2]
  
  if(is.null(Z) == TRUE)
  {
    
    p_cov = 0
    
  }else
  {
    
    p_cov = dim(Z)[2]
    
  }
  
  #### B-Splines for computation ####
  
  if(me_integral_constraint == TRUE)
  {
    
    nspl_ME = K_ME + 4
    
  }else
  {
    
    nspl_ME = K_ME + 3
    
  }
  
  nspl_IE = K_IE + 3
  
  ME_list = array(0, dim = c(n, nspl_ME, p))
  IE_list = array(0, dim = c(n, nspl_IE, p))
  ME_subtract = matrix(0, nrow = p, ncol = nspl_ME)
  
  S_ME_inv = array(0, dim = c(nspl_ME, nspl_ME, p))
  S_IE_inv = array(0, dim = c(nspl_IE, nspl_IE, p))
  
  ME_knots_stor = matrix(0, nrow = p, ncol = K_ME)
  IE_knots_stor = matrix(0, nrow = p, ncol = K_IE)
  
  ind = 1
  
  for(ind in 1:p)
  {
    
    ### For main effects ###
    
    #me_knots = seq(1/M, 1-(1/M), length.out = M)
    
    quantile_seq_ME = seq(0, 1, by = 1/(K_ME+1))
    quantile_seq_ME = quantile_seq_ME[-c(1,K_ME+2)]
    
    me_knots = quantile(X[,ind], quantile_seq_ME)
    
    ME_knots_stor[ind, ] = me_knots
    
    if(me_integral_constraint == TRUE)
    {
    
      me_spl = bSpline(x = X[,ind], knots = me_knots, intercept = TRUE)
      ME_subtract[ind,] = colMeans(me_spl)
      final_Xmat_ME = sweep(me_spl, 2,  ME_subtract[ind,])
      
    
    }else
    {
    
      me_spl = bSpline(x = X[,ind], knots = me_knots, intercept = FALSE)
      final_Xmat_ME = me_spl
      
    }
    
    ME_list[,,ind] = final_Xmat_ME
    
    S_ME_inv[,,ind] = penmatt(nspl_ME)
    
    ### For interaction effects ###
    
    quantile_seq_IE = seq(0, 1, by = 1/(K_IE+1))
    quantile_seq_IE = quantile_seq_IE[-c(1,K_IE+2)]
    
    ie_knots = quantile(X[,ind], quantile_seq_IE)
    
    IE_knots_stor[ind, ] = ie_knots
    
    ie_spl = bSpline(x = X[,ind], knots = ie_knots, intercept = FALSE)
    
    final_Xmat_IE = ie_spl
    
    IE_list[,,ind] = final_Xmat_IE
    
    S_IE_inv[,,ind] = penmatt(nspl_IE) 
    
  }
  
  ME_mat_MEs = NULL
  for(ind in 1:p)
  {
    
    ME_mat_MEs = cbind(ME_mat_MEs, ME_list[,,ind])
    
  }
  
  ME_mat = cbind(rep(1,n), Z, ME_mat_MEs)
  
  map_k_to_uv = matrix(0, nrow = choose(p,2), ncol = 3)
  
  for(k in 1:choose(p,2))
  {
    
    #Obtain inverse index (u,v)
    
    quad_dis = (2*p - 1)^2 - 8*k
    u = ceiling(0.5*((2*p-1) - quad_dis^0.5))
    v = p + k - (u*(p - 0.5*(u+1)))
    
    map_k_to_uv[k,1] = k
    map_k_to_uv[k,2] = u
    map_k_to_uv[k,3] = v
    
  }
  
  map_k_to_uv = map_k_to_uv - 1
  
  SigmaME = solve(penmatt(nspl_ME))
  SigmaME_inv = penmatt(nspl_ME)
  SigmaInt = solve(penmatt(nspl_IE))
  SigmaInt_inv = penmatt(nspl_IE)
  
  # SigmaME = diag(nspl_ME)
  # SigmaME_inv = diag(nspl_ME)
  # SigmaInt = diag(nspl_IE)
  # SigmaInt_inv = diag(nspl_IE)
  
  print(noquote(paste("########## Sampling initiated with MC = ", MC, " ########## ", sep = "")))
  
  SIM_model = SIDsampler_draws_adaptive_optimized(y, 
                                                  ME_mat, 
                                                  IE_list,
                                                  eps_MALA, 
                                                  c_HMC, 
                                                  L_HMC, 
                                                  MC,
                                                  n, 
                                                  p, 
                                                  p_cov,
                                                  SigmaME, 
                                                  SigmaME_inv,
                                                  SigmaInt, 
                                                  SigmaInt_inv,
                                                  nspl_ME, 
                                                  nspl_IE,
                                                  cutoff,
                                                  map_k_to_uv,
                                                  zero_ind,
                                                  accept_low,
                                                  accept_high,
                                                  accept_scale,
                                                  a_lamb,
                                                  b_lamb)
  
  print(noquote(paste("########## Sampling completed with MC = ", MC, " ########## ", sep = "")))
  
  return(list("SIM_model" = SIM_model, 
              "ME_list" = ME_list,
              "ME_knots" = ME_knots_stor,
              "IE_knots" = IE_knots_stor,
              "data" = list("y" = y, "X" = X, "Z" = Z, "n" = n, "p" = p, "MC" = MC)))
  
}

