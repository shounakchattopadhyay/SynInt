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

### Univariate Penalty

penmatt<-function(M)
{
  
  #### Defines univariate penalty matrix. ####
  #### Based on first order differences. ####
  
  matt = matrix(0, nrow = M, ncol = M)
  
  diag(matt)[1:(M-1)] = 2
  diag(matt)[M] = 1
  
  for(i in 1:M)
  {
    for(j in 1:M)
    {
      if(i == j+1)
      {
        matt[i,j] = -1
      }
      else if(j == i+1)
      {
        matt[i,j] = -1
      }
    }
  }
  
  return(matt)
  
}

##M+4 = #splines for main effects, N+3 = #splines for interaction tensor products

SIMsampler<-function(y,
                     X, 
                     Z, 
                     K_ME = 5,
                     K_IE = 2, 
                     eps_MALA = rep(0.01, choose(dim(X)[2], 2)),
                     c_HMC = 1, 
                     L_HMC = 5, 
                     MC = 10000){
  
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
  p_cov = dim(Z)[2]
  
  #### Covariate Effects ####
  
  var_cov = solve(t(Z) %*% Z)
  
  #### B-Splines for computation ####
  
  ME_list = array(0, dim = c(n, K_ME+4, p))
  IE_list = array(0, dim = c(n, K_IE+3, p))
  ME_subtract = matrix(0, nrow = p, ncol = K_ME+4)
  
  S_ME_inv = array(0, dim = c(K_ME+4,K_ME+4,p))
  S_IE_inv = array(0, dim = c(K_IE+3,K_IE+3,p))
  
  ind = 1
  
  for(ind in 1:p)
  {
    
    ### For main effects ###
    
    #me_knots = seq(1/M, 1-(1/M), length.out = M)
    
    quantile_seq_ME = seq(0, 1, by = 1/(K_ME+1))
    quantile_seq_ME = quantile_seq_ME[-c(1,K_ME+2)]
    
    me_knots = quantile(X[,ind], quantile_seq_ME)
    
    me_spl = bSpline(x = X[,ind], knots = me_knots, intercept = TRUE)
    
    ME_subtract[ind,] = colMeans(me_spl)
    final_Xmat_ME = sweep(me_spl, 2,  ME_subtract[ind,])
    
    ME_list[,,ind] = final_Xmat_ME
    
    S_ME_inv[,,ind] = penmatt(K_ME + 4)
    
    ### For interaction effects ###
    
    quantile_seq_IE = seq(0, 1, by = 1/(K_IE+1))
    quantile_seq_IE = quantile_seq_IE[-c(1,K_IE+2)]
    
    ie_knots = quantile(X[,ind], quantile_seq_IE)
    
    ie_spl = bSpline(x = X[,ind], knots = ie_knots, intercept = FALSE)
    
    final_Xmat_IE = ie_spl
    
    IE_list[,,ind] = final_Xmat_IE
    
    S_IE_inv[,,ind] = penmatt(K_IE + 3) 
    
  }
  
  ME_mat = NULL
  for(ind in 1:p)
  {
    
    ME_mat = cbind(ME_mat, ME_list[,,ind])
    
  }
  
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
  
  SigmaME = solve(penmatt(K_ME+4))
  SigmaME_inv = penmatt(K_ME+4)
  SigmaInt = solve(penmatt(K_IE+3))
  SigmaInt_inv = penmatt(K_IE+3)
  
  SIM_model = SIDsampler_draws_adaptive_optimized(y, 
                                                  Z, 
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
                                                  K_ME+4, 
                                                  K_IE+3,
                                                  var_cov, 
                                                  MC,
                                                  map_k_to_uv)
  
  return(SIM_model)
  
}

