#include <RcppArmadillo.h>
#include <Rcpp.h>

// [[Rcpp::depends(RcppArmadillo)]]
//Rcpp::plugins(openmp)

using namespace arma;

// [[Rcpp::export]]
double random_gamma(double a) {
  return R::rgamma(a, 1.0);
}

// [[Rcpp:export]]
double sigmasq_sampler(arma::vec R, int n) {
  
  double sigma_sq = 1.0;
  
  double beta_sigsq = accu(square(R))/2;
  double alpha_sigsq = (n-1) / 2;
  
  sigma_sq = beta_sigsq / random_gamma(alpha_sigsq);
  return sigma_sq;
  
}

// [[Rcpp::export]]
arma::vec maineffects_sampler(arma::vec R, 
                              arma::mat X, 
                              arma::mat Psi_inv, 
                              double sigma_sq){
  
  arma::mat M1 = ((X.t()) * X)/sigma_sq;
  arma::mat M2 = chol(Psi_inv + M1);
  arma::mat M2inv = inv(M2);
  arma::mat betaVar = M2inv * (M2inv.t());
  
  arma::vec betaMean = betaVar * ((X.t() * R)/sigma_sq);
  
  arma::mat sampled_ME_coeff = mvnrnd(betaMean, betaVar, 1);
  return sampled_ME_coeff.col(0);
  
}

// [[Rcpp::export]]
double pot_MALA(arma::vec R, 
                arma::mat X1, 
                arma::mat X2, 
                arma::vec param, 
                double tau1sq, 
                double tau2sq, 
                arma::mat S1, 
                arma::mat S2, 
                double sigma_sq){
  
  int M = X1.n_cols;
  int n = X1.n_rows;
  
  arma::vec theta1 = param(span(0, M-1));
  arma::vec theta2 = param(span(M, (2*M)- 1));
  arma::vec phi1 = param(span((2*M), (3*M)-1));
  arma::vec phi2 = param(span((3*M), (4*M) - 1));
  
  double kappa = param((4*M));
  double pen_param = exp(kappa);
    
  arma::vec mainpart11 = X1 * theta1;
  arma::vec mainpart12 = X2 * phi1;
  arma::vec mainpart21 = X1 * theta2;
  arma::vec mainpart22 = X2 * phi2;
  
  arma::vec Pfn1 = square(mainpart11);
  arma::vec Pfn2 = square(mainpart12);
  arma::vec Pfn = Pfn1 % Pfn2;
    
  arma::vec Nfn1 = square(mainpart21);
  arma::vec Nfn2 = square(mainpart22);
  arma::vec Nfn = Nfn1 % Nfn2;
    
  arma::vec h = Pfn - Nfn;
  
  double pot_lik = accu(square(R - h))/(2*sigma_sq);
  
  arma::mat c_11mat = 0.5*(theta1.t() * (S1 * theta1));
  double c_11 = c_11mat(0,0);
  arma::mat c_21mat = 0.5*(theta2.t() * (S1 * theta2));
  double c_21 = c_21mat(0,0);
  arma::mat c_12mat = 0.5*(phi1.t() * (S2 * phi1));
  double c_12 = c_12mat(0,0);
  arma::mat c_22mat = 0.5*(phi2.t() * (S2 * phi2));
  double c_22 = c_22mat(0,0);
  
  // penalty_term = mean(mainpart1^2 * mainpart2^2)
  // penalty_term = mean((Pfn + Nfn)^2)
  // penalty_term = mean((mainpart1^2) + (mainpart2^2) + (mainpart1^2 * mainpart2^2))
  // penalty_term = mean(Pfn1) * mean(Pfn2) * mean(Nfn1) * mean(Nfn2)
    
  double penalty_term = mean(Pfn) * mean(Nfn);
  
  // penalty_term = (mean(Pfn * Nfn))^2 / pen_param
      
  double pot_prior = ((c_11+c_12) / tau1sq) + 
              ((c_21+c_22) / tau2sq) + 
              (0.5*log(1 + (penalty_term / pen_param))) +
              (pen_param + (0.5*log(pen_param))); 
        
  // Last line = prior for kappa = log(kappa)
        
  double pot_total = pot_lik + pot_prior;
      
  return pot_total;
  
}

// [[Rcpp::export]]
arma::vec grad_MALA(arma::vec R, 
                    arma::mat X1, 
                    arma::mat X2, 
                    arma::vec param, 
                    double tau1sq, 
                    double tau2sq, 
                    arma::mat S1, 
                    arma::mat S2, 
                    double sigma_sq){
  
  int M = X1.n_cols;
  int n = X1.n_rows;
  
  arma::vec theta1 = param(span(0, M-1));
  arma::vec theta2 = param(span(M, (2*M)- 1));
  arma::vec phi1 = param(span((2*M), (3*M)-1));
  arma::vec phi2 = param(span((3*M), (4*M) - 1));
  
  double kappa = param((4*M));
  double pen_param = exp(kappa);
  
  arma::vec mainpart11 = X1 * theta1;
  arma::vec mainpart12 = X2 * phi1;
  arma::vec mainpart21 = X1 * theta2;
  arma::vec mainpart22 = X2 * phi2;
  
  arma::vec Pfn1 = square(mainpart11);
  arma::vec Pfn2 = square(mainpart12);
  arma::vec Pfn = Pfn1 % Pfn2;
  
  arma::vec Nfn1 = square(mainpart21);
  arma::vec Nfn2 = square(mainpart22);
  arma::vec Nfn = Nfn1 % Nfn2;
  
  arma::vec h = Pfn - Nfn;
  
  // Gradient from likelihood
  
  arma::vec v = h - R;
    
  arma::vec grad_lik_part11 = X1.t() * (Pfn2 % (mainpart11 % v));
  arma::vec grad_lik_part21 = - X1.t() * (Nfn2 % (mainpart21 % v));
  arma::vec grad_lik_part12 = X2.t() * (Pfn1 % (mainpart12 % v));
  arma::vec grad_lik_part22 = - X1.t() * (Nfn1 % (mainpart22 % v));
  
  arma::vec grad_lik = (2/sigma_sq)*join_cols(grad_lik_part11,
                                        grad_lik_part21,
                                        grad_lik_part12,
                                        grad_lik_part22);
    
  arma::vec grad_prior = join_cols((S1 * theta1) / tau1sq,
                             (S1 * theta2) / tau2sq,
                             (S2 * phi1) / tau1sq,
                             (S2 * phi2) / tau2sq);
    
 //penalty_term = mean(mainpart1^2 * mainpart2^2)

// # grad_prior_penalty = (2/n)*
// #   c(kronecker(diag(M), t(phi1)) %*% t(X) %*% 
// #       (mainpart1 * Nfn),
// #     kronecker(diag(M), t(phi2)) %*% t(X) %*% 
// #       (mainpart2 * Pfn),
// #     kronecker(t(lambda1), diag(M)) %*% t(X) %*% 
// #       (mainpart1 * Nfn),
// #     kronecker(t(lambda2), diag(M)) %*% t(X) %*% 
// #       (mainpart2 * Pfn))
    
// # penalty_term = mean((Pfn + Nfn)^2)
// # 
// # grad_prior_penalty = (4/n)*
// #   c(t(X1) %*% ((Pfn + Nfn) * Pfn2 * mainpart11),
// #     t(X1) %*% ((Pfn + Nfn) * Nfn2 * mainpart21),
// #     t(X2) %*% ((Pfn + Nfn) * Pfn1 * mainpart12),
// #     t(X2) %*% ((Pfn + Nfn) * Nfn1 * mainpart22))
//     
// # penalty_term = mean((mainpart1^2) + (mainpart2^2) + (mainpart1^2 * mainpart2^2))
// # grad_prior_penalty = (2/n)*c(t(X) %*% (mainpart1 * (1 + Nfn)),
// #                                      t(X) %*% (mainpart2 * (Pfn + 1)))
    
  double penalty_term = mean(Pfn) * mean(Nfn);
      
  double int_P1 = mean(Pfn1);
  double int_P2 = mean(Pfn2);
  double int_N1 = mean(Nfn1);
  double int_N2 = mean(Nfn2);
      
  arma::mat mat1 = (X1.t() * X1) / n ;
  arma::mat mat2 = (X2.t() * X2) / n ;
      
  arma::vec grad_prior_penalty = 2 * join_cols((int_P2 * int_N1 * int_N2) * (mat1 * theta1),
                                     (int_P1 * int_P2 * int_N2) * (mat1 * theta2),
                                     (int_P1 * int_N1 * int_N2) * (mat2 * phi1),
                                     (int_P1 * int_P2 * int_N1) * (mat1 * phi2));
      
// # penalty_term = (mean(Pfn * Nfn))^2
// # grad_prior_penalty = (2/n^2)*c(2*sum(Pfn*Nfn)*t(X) %*% (mainpart1 * Nfn),
// #                                sum(Pfn*Nfn)*t(X) %*% Pfn)
      
  arma::vec grad_total = grad_lik + grad_prior + 
               (0.5*grad_prior_penalty / (pen_param + penalty_term));
        
// Construct gradient wrt pen_param

  double grad_pen = pen_param + ((0.5/(1.0 + (penalty_term / pen_param))));
  arma::vec grad_pen_vec(1, fill::zeros);
  grad_pen_vec(0) = grad_pen;
          
  return join_cols(grad_total, grad_pen_vec);
  
}

//[[Rcpp::export]]
Rcpp::List sq_sampler(arma::vec R, 
                      arma::mat X1, 
                      arma::mat X2, 
                      double tau1sq, 
                      double tau2sq, 
                      arma::mat S1, 
                      arma::mat S2, 
                      double sigma_sq,
                      arma::vec old_param, 
                      double eps_MALA,
                      double c_HMC, 
                      int L_HMC){
  
  int M = old_param.n_elem;
  
  arma::vec new_param = old_param;
  arma::mat sig1;
  sig1.eye(M,M);
  arma::vec new_rho = mvnrnd(vec(M, fill::zeros), sig1, 1);
  
  arma::vec current_param = new_param;
  arma::vec current_rho = new_rho;
  
  //Half step of momentum
  
  new_rho = new_rho - (0.5*eps_MALA*grad_MALA(R, X1, X2,
                                              new_param, tau1sq, tau2sq,
                                              S1, S2, sigma_sq));
  
  for(int l_ind=0; l_ind < L_HMC; ++l_ind){
    
    //Full position step
    
    new_param = new_param + ((eps_MALA * pow(c_HMC, 2.0)) * new_rho);
    
    if(l_ind != L_HMC - 1){
      
      //Full Momentum Step
      
      new_rho = new_rho - (eps_MALA*grad_MALA(R, X1, X2,
                                              new_param, tau1sq, tau2sq,
                                              S1, S2, sigma_sq));
      
    }
    
  }
  
  //Half momentum step
  
  new_rho = new_rho - (0.5*eps_MALA*grad_MALA(R, X1, X2,
                                              new_param, tau1sq, tau2sq,
                                              S1, S2, sigma_sq));
  
  new_rho = -new_rho; //To maintain reversibility
  
  double current_U = pot_MALA(R, X1, X2, current_param, tau1sq, tau2sq, 
                              S1, S2, sigma_sq);
  double current_K = sum(square(current_rho))/(2*pow(c_HMC, 2.0));
  
  double new_U = pot_MALA(R, X1, X2, new_param, tau1sq, tau2sq, 
                          S1, S2, sigma_sq);
  double new_K = sum(square(new_rho))/(2*pow(c_HMC, 2.0));
  
  double U1 = randu();
  double energy_diff1 = exp(current_U - new_U + current_K - new_K);
  int sample_accept = 0;
  arma::vec param_accept(M, fill::zeros);
  
  arma::vec ediff(1, fill::zeros);
  ediff(0) = energy_diff1;
  
  if(ediff.is_finite() == TRUE){
    
    if(U1 <= energy_diff1){
      
      param_accept = new_param;
      sample_accept = 1;
      
    }
    else{
      
      param_accept = old_param;
      sample_accept = 0;
      
    }
    //print("No Issues with HMC.")
  }
  else{
    
    param_accept = old_param;
    sample_accept = 0;
    
    //print("HMC Diverged.")
  }
  
  return Rcpp::List::create(Rcpp::Named("sampled_param") = param_accept,
                            Rcpp::Named("accept")      = sample_accept);
  
}

// [[Rcpp::export]]
Rcpp::List SIDsampler_draws_adaptive_optimized(arma::vec y, 
                                               arma::mat Z, 
                                               arma::mat ME_mat, 
                                               arma::cube IE_list,
                                               arma::vec eps_MALA, 
                                               double c_HMC, 
                                               int L_HMC, 
                                               int MC,
                                               int n, 
                                               int p, 
                                               int p_cov,
                                               arma::mat SigmaME, 
                                               arma::mat SigmaME_inv,
                                               arma::mat SigmaInt, 
                                               arma::mat SigmaInt_inv,
                                               int ME_nspl, 
                                               int IE_nspl,
                                               arma::mat var_cov, 
                                               int cutoff,
                                               arma::mat map_k_to_uv,
                                               arma::vec zero_ind,
                                               double accept_low,
                                               double accept_high,
                                               double accept_scale){
  //Define storage matrices
  
  arma::vec alpha_stor(MC, fill::zeros);
  arma::vec sigmasq_stor(MC, fill::ones);
  
  arma::mat ME_coeff_stor(MC, ME_nspl*p, fill::zeros);
  arma::mat ME_scale_stor(MC, p, fill::ones);
  arma::mat ME_scale_aux(MC, p, fill::ones);
  
  int K = p*(p-1)/2;
  // int IE_nbasis = IE_nspl * IE_nspl;
  
  arma::cube IE_pos_theta1(MC, IE_nspl, K, fill::zeros);
  arma::cube IE_neg_theta2(MC, IE_nspl, K, fill::zeros);
  arma::cube IE_pos_phi1(MC, IE_nspl, K, fill::zeros);
  arma::cube IE_neg_phi2(MC, IE_nspl, K, fill::zeros);
  
  arma::mat IE_scale_tausq1(MC, K, fill::ones);
  arma::mat IE_scale_tausq2(MC, K, fill::ones);
  arma::mat IE_scale_a(MC, K, fill::ones);
  arma::mat IE_scale_b(MC, K, fill::ones);
  
  arma::mat IE_pen(MC, K, fill::ones);
  
  arma::mat all_interactions(n, K, fill::zeros);
  
  arma::mat cov_effect_stor(MC, p_cov, fill::zeros);
  
  for(int k=0; k<K; ++k){
    
    // Obtain inverse index (u,v)
    
    int u = map_k_to_uv(k,1);
    int v = map_k_to_uv(k,2);
      
    // Initialize rank 1 components
    
    if(zero_ind(k) == 1){
      
      (IE_pos_theta1.slice(k)).row(0) = mvnrnd(arma::vec(IE_nspl, fill::zeros),
       0.1*SigmaInt).t();
      (IE_neg_theta2.slice(k)).row(0) = mvnrnd(arma::vec(IE_nspl, fill::zeros),
       0.1*SigmaInt).t();
      (IE_pos_phi1.slice(k)).row(0) = mvnrnd(arma::vec(IE_nspl, fill::zeros),
       0.1*SigmaInt).t();
      (IE_neg_phi2.slice(k)).row(0) = mvnrnd(arma::vec(IE_nspl, fill::zeros),
       0.1*SigmaInt).t();
      
      arma::vec pos_part1 = square(IE_list.slice(u) * IE_pos_theta1.slice(k).row(0).t());
      arma::vec pos_part2 = square(IE_list.slice(v) * IE_pos_phi1.slice(k).row(0).t());
      arma::vec neg_part1 = square(IE_list.slice(u) * IE_neg_theta2.slice(k).row(0).t());
      arma::vec neg_part2 = square(IE_list.slice(v) * IE_neg_phi2.slice(k).row(0).t());
      
      arma::vec pos_part = pos_part1 % pos_part2;
      arma::vec neg_part = neg_part1 % neg_part2;
      
      all_interactions.col(k) = pos_part - neg_part;
      
    }
    
  }
  
  arma::mat accept_MALA(MC, K, fill::zeros);
  accept_MALA.row(0) = vec(K, fill::ones).t();
  
  //Begin MCMC sampling. 
  
  for(int m=1; m<MC; ++m){
    
    // //Sample (\alpha, \sigma^2) | - 
    // 
    // arma::vec R_int_sigsq = y - ((ME_mat * ME_coeff_stor.row(m-1).t()) +
    //                       (sum(all_interactions, 1)) +
    //                       (Z * cov_effect_stor.row(m-1).t()));
    // 
    // sigmasq_stor(m) = sigmasq_sampler(R_int_sigsq, n);
    // alpha_stor(m) = intercept_sampler(R_int_sigsq, n, sigmasq_stor(m));
    
    ////Main Effects and Related Parameters (intercept, sigma^2)
    
    arma::vec R_ME = y - sum(all_interactions, 1);
    
    arma::mat mat_ME_scales(p, p, fill::zeros);
    mat_ME_scales.diag() = ME_scale_stor.row(m-1).t();
    
    arma::mat Psi_ME_inv = kron(mat_ME_scales, SigmaME_inv);
    
    arma::mat whole_prec(1+p_cov+(p*ME_nspl), 1+p_cov+(p*ME_nspl), fill::zeros);
    whole_prec(0,0) = 0.01;
    whole_prec.submat(1, 1, p_cov, p_cov).eye();
    whole_prec.submat(p_cov+1, p_cov+1, ((p*ME_nspl)+p_cov), ((p*ME_nspl)+p_cov)) = Psi_ME_inv;
    
    arma::vec sampler_ME = maineffects_sampler(R_ME, 
                                         ME_mat, 
                                         whole_prec,
                                         sigmasq_stor(m-1));
    
    alpha_stor(m) = sampler_ME(0);
    cov_effect_stor.row(m) = sampler_ME(span(1, p_cov)).t();
    ME_coeff_stor.row(m) = sampler_ME(span(p_cov+1,p_cov+(p*ME_nspl))).t();
    
    //Sample \lambda_j = scale of \beta_j ####
    
    arma::vec qf_stor(p, fill::ones);
    
    for(int j=0; j<p; ++j){
      
      arma::vec beta_j = ME_coeff_stor(m, span(j*ME_nspl, ((j+1)*ME_nspl)-1)).t();
      arma::mat qf_j = beta_j.t() * (SigmaME_inv * beta_j);
      qf_stor(j) = qf_j(0,0);
      
      ME_scale_stor(m,j) = random_gamma(0.5 + 
                          (0.5*ME_nspl))/(ME_scale_aux(m-1,j) + (0.5*qf_stor(j)));
      
      ME_scale_aux(m,j) = random_gamma(1.0) / (1 + ME_scale_stor(m,j));
      
    }
    
    //Interaction Effects and Related Parameters. #####
    
    for(int k=0; k<K; ++k){
      
      if(zero_ind(k) == 1){
        
        // Obtain inverse index (u,v)
        
        int u = map_k_to_uv(k,1);
        int v = map_k_to_uv(k,2);
        
        // Remove index k and sum up other interactions
        
        arma::mat dummy_all_int = all_interactions;
        dummy_all_int.shed_col(k);
        arma::vec all_interaction_sum_notk = sum(dummy_all_int, 1);
        
        // Sample interaction k
        
        arma::vec R_IE_k = y - ((ME_mat * sampler_ME) +
                               (all_interaction_sum_notk));
        
        // Define old_param for sampling
        
        arma::vec old_param((4*IE_nspl + 1), fill::zeros);
        
        old_param(span(0, IE_nspl - 1)) = IE_pos_theta1.slice(k).row(m-1).t();
        old_param(span(IE_nspl, (2*IE_nspl) - 1)) = IE_neg_theta2.slice(k).row(m-1).t();
        old_param(span(2*IE_nspl, (3*IE_nspl)-1)) = IE_pos_phi1.slice(k).row(m-1).t();
        old_param(span(3*IE_nspl, (4*IE_nspl)-1)) = IE_neg_phi2.slice(k).row(m-1).t();
        old_param(4*IE_nspl) = log(IE_pen(m-1,k));
        
        // Do the sampling
        
        Rcpp::List sq_MALA_k = sq_sampler(R_IE_k, 
                                          IE_list.slice(u), 
                                          IE_list.slice(v), 
                                          IE_scale_tausq1(m-1,k),
                                          IE_scale_tausq2(m-1,k),
                                          SigmaInt_inv,
                                          SigmaInt_inv,
                                          sigmasq_stor(m-1),
                                          old_param,
                                          eps_MALA(k),
                                          c_HMC,
                                          L_HMC);
        
        accept_MALA(m,k) = sq_MALA_k["accept"];
        arma::vec whole_coeff = sq_MALA_k["sampled_param"];
        int len_psi = (whole_coeff.n_elem - 1)/4;
        arma::vec whole_coeff_psi = whole_coeff(span(0, 4*len_psi - 1));
        
        IE_pos_theta1.slice(k).row(m) = whole_coeff_psi(span(0, len_psi-1)).t();
        IE_neg_theta2.slice(k).row(m) = whole_coeff_psi(span(len_psi, 2*len_psi-1)).t();
        IE_pos_phi1.slice(k).row(m) = whole_coeff_psi(span(2*len_psi, 3*len_psi-1)).t();
        IE_neg_phi2.slice(k).row(m) = whole_coeff_psi(span(3*len_psi, 4*len_psi-1)).t();
        IE_pen(m,k) = exp(whole_coeff(4*len_psi));
        
        // Update the interaction matrix
        
        arma::vec Ppart1 = square(IE_list.slice(u) * IE_pos_theta1.slice(k).row(m).t());
        arma::vec Ppart2 = square(IE_list.slice(v) * IE_pos_phi1.slice(k).row(m).t());
        arma::vec Npart1 = square(IE_list.slice(u) * IE_neg_theta2.slice(k).row(m).t());
        arma::vec Npart2 = square(IE_list.slice(v) * IE_neg_phi2.slice(k).row(m).t());
        
        arma::vec Ppart = Ppart1 % Ppart2;
        arma::vec Npart = Npart1 % Npart2;
        
        all_interactions.col(k) = Ppart - Npart; 
        
        // Update other parameters
        
        arma::mat c_1k1_mat = 0.5*(IE_pos_theta1.slice(k).row(m) * (SigmaInt_inv *
          IE_pos_theta1.slice(k).row(m).t()));
        double c_1k1 = c_1k1_mat(0,0);
        
        arma::mat c_1k2_mat = 0.5*(IE_pos_phi1.slice(k).row(m) * (SigmaInt_inv *
          IE_pos_phi1.slice(k).row(m).t()));
        double c_1k2 = c_1k2_mat(0,0);
        
        double c_1k = c_1k1 + c_1k2;
        
        arma::mat c_2k1_mat = 0.5*(IE_neg_theta2.slice(k).row(m) * (SigmaInt_inv *
          IE_neg_theta2.slice(k).row(m).t()));
        double c_2k1 = c_2k1_mat(0,0);
        
        arma::mat c_2k2_mat = 0.5*(IE_neg_phi2.slice(k).row(m) * (SigmaInt_inv *
          IE_neg_phi2.slice(k).row(m).t()));
        double c_2k2 = c_2k2_mat(0,0);
        
        double c_2k = c_2k1 + c_2k2;
        
        // Sample \tau_1^2 | a and a | \tau_1^2
        
        double tausq1_rate = (1.0/IE_scale_a(m-1,k)) + c_1k;
        
        IE_scale_tausq1(m,k) = tausq1_rate / random_gamma(IE_nspl + 0.5);
        IE_scale_a(m,k) = (1 + (1/IE_scale_tausq1(m,k))) / random_gamma(1.0);
        
        // Sample \tau_2^2 | b and b | \tau_2^2
        
        double tausq2_rate = (1.0/IE_scale_b(m-1,k)) + c_2k;
        
        IE_scale_tausq2(m,k) = tausq2_rate / random_gamma(IE_nspl + 0.5);
        IE_scale_b(m,k) = (1 + (1/IE_scale_tausq2(m,k))) / random_gamma(1.0);
        
      }
      
      // Sample error variance
      
      arma::vec R_sigsq = y - ((ME_mat * sampler_ME) + sum(all_interactions,1));
      sigmasq_stor(m) = sigmasq_sampler(R_sigsq, n);
      
    }
    
    if((m >= 199) && (m < cutoff)){
      
      if(m % 100 == 0){
        
        for(int k1=0; k1<K; ++k1){
          
          if(mean(accept_MALA.rows(m-199,m).col(k1)) <= accept_low){
            
            eps_MALA(k1) = eps_MALA(k1)*accept_scale;
            
          }else if(mean(accept_MALA.rows(m-199,m).col(k1)) >= accept_high){
            
            eps_MALA(k1) = eps_MALA(k1)/accept_scale;
            
          }else{
            
            eps_MALA(k1) = eps_MALA(k1);
            
          }
          
        }
        
      }
      
    }  
    
    if((m+1) % 1000 == 0){
      
      Rcpp::Rcout << "Monte Carlo Sample: " << m+1 << std::endl;
    
    }
    
    Rcpp::Rcout << "MCMC Iterate: " << m << std::endl;
    // Rcpp::Rcout << "Accept Ratio: " << mean(accept_MALA.rows(0,m), 0) << std::endl;
    // Rcpp::Rcout << "Current step-size: " << eps_MALA.t() << std::endl;
    
  }  
  
  // #### Special case p = 2 ####
  //                         
  //                         if(m %% 200 == 0)
  //                         {
  //                           
  //                           if(max(mean(accept_MALA[(m-199):m])) <= 0.5)
  //                           {
  //                             
  //                             eps_MALA = 0.9*eps_MALA
  //                             
  //                           }
  //                           else if(min(mean(accept_MALA[(m-199):m])) >= 0.9)
  //                           {
  //                             
  //                             eps_MALA = eps_MALA/0.9
  //                             
  //                           }else
  //                           {
  //                             
  //                             eps_MALA = eps_MALA
  //                             
  //                           }
  //                           
  //                         }
  //                         
  //         }
  
  return Rcpp::List::create(Rcpp::Named("intercept") = alpha_stor,
                            Rcpp::Named("error_var") = sigmasq_stor,
                            Rcpp::Named("ME_coeff") = ME_coeff_stor,
                            Rcpp::Named("ME_scale") = ME_scale_stor,
                            Rcpp::Named("IE_pos_coeff_part1") = IE_pos_theta1,
                            Rcpp::Named("IE_pos_coeff_part2") = IE_pos_phi1,
                            Rcpp::Named("IE_neg_coeff_part1") = IE_neg_theta2,
                            Rcpp::Named("IE_neg_coeff_part2") =  IE_neg_phi2,
                            Rcpp::Named("cov_effects") = cov_effect_stor,
                            Rcpp::Named("ME_mat") = ME_mat,
                            Rcpp::Named("IE_list") = IE_list,
                            Rcpp::Named("Accept_Prop") = accept_MALA,
                            Rcpp::Named("HMC_epsilon") = eps_MALA);
  
}
