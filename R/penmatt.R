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