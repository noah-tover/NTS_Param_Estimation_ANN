library(parallel)
library(doParallel)
library(foreach)
library(temStaR)
library(RcppCNPy)
library(matrixStats)
library(reticulate)
################################################################################
#' @description This version of chf_stdNTS is significantly faster by avoiding checks and indexes - parameters are explicitly defined. This function was not replaced because there were significant dependencies relying on the vector containing the parameters.
chf_stdNTS_fast <- function(u, alpha, theta, beta, gamma, dt){
  m  = 0;
  y = exp( dt*( 1i*(m-beta)*u - 2*theta^(1-alpha/2)/alpha*((theta-1i*(beta*u+1i*gamma^2*u^2/2))^(alpha/2)-theta^(alpha/2) ) ) )
  return( y )
}

################################################################################
#' @description Generates a series of risk-neutral asset prices under stdNTS 
#' with *constant* beta,gamma, and a GARCH(1,1)-type sigma path. This is 
#' @param alpha Alpha parameter of stdNTS distribution.
#' @param theta Theta parameter of stdNTS distribution.
#' @param beta Constant beta parameter for stdNTS.
#' @param gamma Constant gamma parameter for stdNTS.
#' @param kappa GARCH(1,1) parameter.
#' @param xi    GARCH(1,1) parameter.
#' @param lambda GARCH(1,1) parameter (constant).
#' @param zeta  GARCH(1,1) parameter.
#' @param sigma0 Initial value for sigma in GARCH(1,1).
#' @param S0  Initial asset price.
#' @param y0  Initial log-return (default 0).
#' @param npath Number of simulation paths.
#' @param ntimestep Number of timesteps.
#' @param dt  Time step size in years (default 1/250).
#' @param r   Risk-free rate per step (default 1/250).
#' @param d   Dividend yield (default 0).
#' @return A matrix of dimension npath x ntimestep of simulated asset prices.
################################################################################
genrfsamplestdNTSprices <- function(alpha, theta, beta, gamma,
                                    kappa, xi, lambda, zeta, 
                                    sigma0, S0, y0 = 0, 
                                    npath, ntimestep,
                                    dt = 1/250, r = 1/250, d = 0, index){
  # 1) Generate stdNTS errors (constant beta, gamma)
  
  ntsparam <- list(alpha, theta, beta, gamma, 0.0)
  error_np <- rnts_gpu_module$rnts(reticulate::tuple(as.integer(npath), as.integer(ntimestep)), ntsparam)
  error <- py_to_r(error_np)
 # GPU things much better in python
  # Ensure dimensions match
  if (!all(dim(error) == c(npath, ntimestep))) {
    stop("Dimensions of the error matrix do not match npath and ntimestep.")
  }
  
  
  sigma <- gensamplesigmapaths(error, kappa, xi, lambda, zeta, sigma0)
  
  # 3) Generate log-return paths
  paths <- matrix(0, nrow = npath, ncol = ntimestep)
  w <- log(chf_stdNTS_fast(u = -1i * sigma, alpha = alpha, beta = beta, theta = theta, gamma = gamma, dt = 1))
  paths <- r - d - w + sigma * error
  paths[, 1] <- y0
  
  # 4) Convert log-returns to prices
  paths <- t(apply(paths, 1, cumsum))
  prices <- S0 * exp(paths)
  prices <- Re(prices)  # remove any imaginary part
  return(prices)
}

gensamplesigmapaths <- function(sample_errors, kappa, xi, lambda, zeta, sigma0){
  paths <- matrix(0, nrow = nrow(sample_errors), ncol = ncol(sample_errors))
  paths[, 1] <- sigma0
  for(i in 1:nrow(paths)){
    for(t in 2:ncol(paths)){
      sigma_today <- paths[i, t-1]
      error_today <- sample_errors[i, t-1]
      sigma_tomorrow <- sqrt(kappa + xi * sigma_today^2 * (error_today - lambda)^2 + zeta * sigma_today^2)
      paths[i, t] <- sigma_tomorrow
    }
  }
  return(paths)
}

################################################################################
#' @description Simple wrapper for computing option prices under stdNTS 
#' with *constant* beta,gamma (plus GARCH(1,1) for volatility). It simulates 
#' @param npath Number of simulation paths for asset price.
#' @param alpha,theta,beta,gamma stdNTS parameters, with beta,gamma fixed.
#' @param kappa,xi,lambda,zeta,sigma0 GARCH(1,1) parameters.
#' @param S0 Initial asset price.
#' @param y0 Initial log-return (default 0).
#' @param pct_otm Vector of OTM percentages for calls/puts. E.g. c(0.1, 0.2).
#' @param t  Maturity in years (e.g. 0.5 means 6-month).
#' @param r  Risk-free rate per step (default 0.2/250).
#' @return A length-2 numeric vector: the mean call and put prices across paths.
################################################################################

stdNTSoption <- function(npath,
                         alpha, theta, 
                         beta, gamma,
                         kappa, xi, lambda, zeta, 
                         sigma0, S0 = 100, y0 = 0,
                         moneyness = 1.5, 
                         tao = 30/250, 
                         r = 0.2/250, index
) {
  
  # Convert continuous time in years to # of steps
  ntimestep <- ceiling(tao * 250)
  
  sample_prices <- genrfsamplestdNTSprices(
    alpha, theta, beta, gamma, 
    kappa, xi, lambda, zeta, sigma0,
    S0, y0, npath, ntimestep, dt = 1/250, r = r, index = index
  )
  
  option_prices <- gensamplerfoptionprices(
    r = r, moneyness = moneyness,
    sample_prices = sample_prices, type = 'European'
  )
  
  # Combine call and put prices into a matrix
  # Convert to values
  result_df <- as.data.frame(option_prices) / S0
  return(t(result_df))
}
gensamplerfoptionprices <- function(r = .02/250, moneyness = 1.5, sample_prices, type = c('US', 'European')) {
  
  # Define the maximum starting time for a t day option. 
  t <- ncol(sample_prices)
  # If we are only interested in a single option path, will only compute a path for one option. Otherwise, calculates a new one every single day,.
  discount_factors <- exp(-r * (0:(t - 1)))
  
  # these whill be lists of matrices storing the option prices at different starting points.
  call_matrix <- matrix(NA, nrow = 1, ncol = t)
  put_matrix  <- matrix(NA, nrow = 1, ncol = t)
  
  
  strike <- sample_prices[, 1] * moneyness
  
  if(type == "European"){
    S_t <- sample_prices[, t]
    call_payoff <- max(S_t - strike, 0)
    put_payoff <- max(strike - S_t, 0)
    call_price <- discount_factors[t] * mean(call_payoff)
    put_price <- discount_factors[t] * mean(put_payoff)
    option_prices <- c(call_price, put_price)
    names(option_prices) <- c('Call', 'Put')
  } else if(type == "US"){
    # j = t is expiration date.
    for (j in 1:t) {
      S_t <- sample_prices[, j] 
      # Calculate the option payoff for each path.
      call_payoff <- pmax(S_t - strike, 0)
      put_payoff  <- pmax(strike - S_t, 0)
      # Discounting and averaging the results of all simulation paths.
      call_matrix[, j] <- discount_factors[j] * mean(call_payoff)
      put_matrix[, j]  <- discount_factors[j] * mean(put_payoff)
    }
    
    call_price <- call_matrix
    put_price <- put_matrix
    option_prices <- list(call_matrix, put_matrix)
    names(option_prices) <- c("Call", "Put")
  }
  
  return(option_prices)
  
  
}


library(parallel)
library(doParallel)
library(foreach)

stdNTSoptionmontecarlo <- function(n_sim, 
                                   chunk_size = 1000, 
                                   output_dir, 
                                   npath = 20000,
                                   r = .02/250, 
                                   ncores = detectCores() - 1
                                   , 
                                   S0 = 1,
                                   y0 = 0,
                                   sigma0 = 0.0096,
                                   nstart = 1) {
  halton <- simulateHaltonVectors(n = n_sim, sim_B = TRUE)
  halton <- halton[nstart:nrow(halton), , drop = FALSE] # Halton vectors are deterministic - so this approach is ok.
  
  
  cl <- makeCluster(ncores)
  registerDoParallel(cl)
  clusterEvalQ(cl, {
    library(reticulate)
    reticulate::use_python("C:/Users/noah/anaconda3/envs/gpu_py/python.exe", required = TRUE)
    
    
    assign(
      "rnts_gpu_module",
      reticulate::import_from_path("rnts_gpu_module", path = "C:/Users/noah/Downloads/mcs_stdNTS_data/mcs_code"),
      envir = .GlobalEnv
    )
  })
  nchunks <- ceiling(nrow(halton) / chunk_size)
  
  cat("Beginning Monte Carlo simulation...")
  
  for (chunk_idx in 1:nchunks) {
    idx_start <- (chunk_idx - 1) * chunk_size + 1
    idx_end <- min(chunk_idx * chunk_size, nrow(halton))
    halton_chunk <- halton[idx_start:idx_end, , drop = FALSE]
    
    results_chunk <- foreach(i = 1:nrow(halton_chunk), .combine = rbind, 
                             .packages = c("temStaR", "randtoolbox", "RcppCNPy"), .export = c("stdNTSoption",
                                                                                              "genrfsamplestdNTSprices",
                                                                                              "gensamplesigmapaths",
                                                                                              "chf_stdNTS_fast",
                                                                                              "gensamplerfoptionprices")) %dopar% {
                                                                                                params <- halton_chunk[i, ]
                                                                                                option_matrix <- stdNTSoption(npath = npath,
                                                                                                                              alpha = as.numeric(params["alpha"]),
                                                                                                                              theta = as.numeric(params["theta"]),
                                                                                                                              kappa = params["kappa"],
                                                                                                                              xi = params["xi"],
                                                                                                                              lambda = params["lambda"],
                                                                                                                              zeta = params["zeta"],
                                                                                                                              sigma0 = sigma0,
                                                                                                                              S0 = S0,
                                                                                                                              y0 = y0,
                                                                                                                              moneyness = params["moneyness"],
                                                                                                                              tao = params["tao"],
                                                                                                                              r = r,
                                                                                                                              beta = as.numeric(params["betas"]),
                                                                                                                              gamma = as.numeric(params["gammas"]),
                                                                                                                              index = as.numeric(params["index"]))
                                                                                                
                                                                                                mcs_row <- merge(t(as.data.frame(params)), as.data.frame(option_matrix))
                                                                                                return(mcs_row)
                                                                                              }
    
    # Save to disk
    filename <- paste0(output_dir, "stdNTSoptionpricemcs_", idx_start + nstart - 1, "_", idx_end + nstart - 1, ".csv")
    data.table::fwrite(results_chunk, filename)
    cat("Saved chunk", chunk_idx, "to:", filename, "\n")
  }
  
  stopCluster(cl)
  cat("Monte Carlo simulation complete :D")
}

library(randtoolbox)
# First simulate halton vectors of parameters.
simulateHaltonVectors <- function(n = 100000, sim_B = FALSE) {
  n = n + 20 # Later removing the first 20 rows to make lower discrepancy from correlation.
  if(sim_B == TRUE){
    dim = 11
  } else {
    dim = 10
  }
  halton_points <- halton(n = n, dim = dim)
  # 1. alpha: Uniform(0,1) -> (0, 2)
  halton_points[, 1] <- 2 * halton_points[, 1]
  
  # 2. theta: Using an exponential decay with mean 1.2544.
  #    qexp(u, rate = 1/mean) converts u ~ U(0,1) into an exponential variable.
  halton_points[, 2] <- qexp(halton_points[, 2], rate = 1 / 1.2544)
  
  # 3. a_1: Uniform(-1, 1), but must not equal 0.
  halton_points[, 3] <- 2 * halton_points[, 3] - 1
  halton_points[, 3][halton_points[, 3] == 0] <- .Machine$double.eps  # adjust any exact 0 to a tiny value
  
  # 4. Moneyness ~ Uniform(.5, 1.5)
  halton_points[,4] <- .5 * halton_points[,4] + .5
  halton_points[,4][halton_points[,4] == 1] <- 1 + runif(1, min = -.5, max = .5) # if moneyness is exactly
  
  # 5. Tao ~ Uniform(.4, 1) 
  halton_points[, 5] <- .6 * halton_points[, 5] + .4
  
  # 6. kappa: Positive with no upper bound.
  #    Again using an exponential transformation (mean = 1).
  halton_points[, 6] <- qexp(halton_points[, 6], rate = 1)
  
  # 7. xi: Uniform(0,1)
  
  # 8. zeta: Uniform(0,1) with the added constraint that zeta + xi < 1.
  #    To ensure this, we let zeta = u * (1 - xi).
  halton_points[, 8] <- halton_points[, 8] * (1 - halton_points[,7])
  
  # 9. sigma_error: Uniform over [-0.05405997595, 0.05405997595] but not 0.
  halton_points[, 9] <- 0.05405997595 * 2 * halton_points[, 9] - 0.05405997595
  halton_points[, 9][halton_points[, 9] == 0] <- .Machine$double.eps  # adjust any exact 0
  
  # 10. Lambda ~ Uniform(0, 0.8)
  halton_points[, 10] <- 0.8 * halton_points[, 10]
  # 11. B ~ Uniform(-1, 1)
  if(sim_B == TRUE){
    halton_points[, 11] <- 2 * halton_points[, 11] - 1
    halton_points[, 11][halton_points[, 11] == 0] <- .Machine$double.eps  # adjust any exact 0 to a tiny value
    beta <- halton_points[, 11] * sqrt(2 * halton_points[,2] / (2 - halton_points[,1]))
    gamma <- (1 - halton_points[,11] ^ 2)
    halton_points <- cbind(halton_points, beta, gamma)
  }
  
  # Drop first 20 rows to avoid correlation problem.
  halton_points <- halton_points[-(1:20),]
  if(sim_B == TRUE){
    colnames(halton_points) <- c("alpha", "theta", "a1", "moneyness", "tao", "kappa", "xi", "zeta", "sigma_error", "lambda", 'B', 'betas', 'gammas')
  } else {
    colnames(halton_points) <- c("alpha", "theta", "a1", "moneyness", "tao", "kappa", "xi", "zeta", "sigma_error", "lambda")
  }
  halton_points <- cbind(index = seq_len(nrow(halton_points)), halton_points)
  return(halton_points)
}


set.seed(1234)
stdNTSoptionmontecarlo(n_sim = 100000, chunk_size = 30, ncores = 4, output_dir = "C:/Users/noah/Downloads/mcs_stdNTS_data/", npath = 20000, nstart = 41508)
