# Note: This code needs to be refactored to match stdNTS - possibly merged (vectorization, gpu rnts, etc..)

#################################################################################################################################################################################
#' @description This function generates sample paths for the stochastic parameter controlling the skewness and kurtosis of the stoT-NTS model. Can also calculate beta and gamma parameters. 
#' @param alpha The alpha parameter of the stdNTS distribution. Assumed to be a static number between 0 and 2, not inclusive.
#' @param theta THe theta parameter of the stdNTS distribution. Assumed to be a static number above 0.
#' @param npath The number of B paths to simulate
#' @param ntimestep The number of time periods to simulate.
#' @param a0 The intercept of the ARMA process governing B_t.
#' @param a1 The slope parameter of the ARMA process governing B_t
#' @param sigma_error The scale parameter of the error term for the ARMA process governing B_t
#' @param B0 The initial value of B.
#' @param calculate_beta_gamma If TRUE, calculates beta and gamma parameters for each B_t path.
#' @return If compute_beta_gamma is false, npath x ntimestep matrix of B paths. If calculate_beta_gamma is true, the result is a list of 3 x ntimestep matrices with each B path and its respective beta and gamma parameters through time at each index.
computebetagamma <- function(B_paths, alpha, theta){
  betas <- B_paths * sqrt(2* theta / (2 - alpha))
  gammas <- sqrt(1 - B_paths^2)
  B_paths <- rbind(B_paths, betas, gammas)
  rownames(B_paths) <- c('B_paths', 'betas', 'gammas')
  return(B_paths)
}
gensampleBpath <- function(alpha, theta, npath, ntimestep, a0, a1, sigma_error, B0, calculate_beta_gamma = FALSE) {
  
  
  # Initialize an empty matrix to store the paths.
  B_paths <- matrix(0, nrow = npath, ncol = ntimestep)
  # Initialize a matrix of errors standard normal errors.
  errors <- matrix(rnorm(npath * ntimestep), nrow = npath, ncol = ntimestep)
  for(i in 1:npath){
    B_paths[i, 1] <- B0
    # B = B_t + DeltaB_t+1. DeltaB_t+1 = a0 + a1 * DeltaB_t + sigma_error * epsilon. Epsilon ~ N(0,1) and i.i.d. 
    for(t in 2:ntimestep){
      delta_B_today <- B_paths[i, t - 1] - (if (t > 2) B_paths[i, t - 2] else B0)
      delta_B_tomorrow <- a0 + a1 * delta_B_today + sigma_error * errors[i, t]
      B_paths[i, t] <- B_paths[i, t - 1] + delta_B_tomorrow

    }
  }
  # TODO: Double check that this is appropriate.
  B_paths[B_paths == 0] <- 1e-10 # Replace any 0s with a trivially small number. The paper does not specify the values of gamma and beta when B_t == 0
  # Ensure B_paths are constrained by -1 < B < 1
  B_paths <- tanh(B_paths)
  if(calculate_beta_gamma == TRUE){
    B_paths <- split(B_paths, row(B_paths))
    result <- lapply(B_paths, computebetagamma, alpha, theta)
  } else {
    result <- B_paths
  }
  
  return(result)
}

#################################################################################################################################################################################

#################################################################################################################################################################################
#' @description Generates a series of stdNTS random numbers with the beta and gamma parameters changing at each t according to their respective B paths. Represents errors of the stoT-NTS model.
#' @param alpha The alpha parameter of the stdNTS distribution. Assumed to be a static number between 0 and 2, not inclusive.
#' @param theta The theta parameter of the stdNTS distribution. Assumed to be a static number above 0.
#' @param npath The number of paths to simulate
#' @param ntimestep The number of timesteps to simulate.
#' @param dt 
#' @param B_params A single resulting matrix of the gensampleBpath function's list with calculate_beta_gamma = TRUE. Only one list item may be passed.
#' @return A npath x ntimestep matrix of stoTntserrors.
gensamplestoTntserrors <- function(alpha, theta, npath, ntimestep, dt, B_params, parallel = TRUE, cluster = NULL) {
  paths <- matrix(0, nrow = npath, ncol = ntimestep)
  u <- pracma::rand(npath, ntimestep)
  if(parallel){
    if (is.null(cluster)) stop("Cluster must be provided when parallel = TRUE")
    registerDoParallel(cluster)
    paths <- foreach(i = 1:npath, .combine = 'rbind', .packages = c("temStaR")) %dopar% {
      worker_vec <- numeric(ntimestep)
      for(t in 2:ntimestep){
        ntsparam <- c(alpha, theta, B_params['betas' ,t], B_params['gammas' ,t])
        u_i <- u[i, t]
        worker_vec[t] <- ipnts(u_i, ntsparam) 
      }
      worker_vec
    }
  } else {
    for (i in 1:npath) {
      for(t in 2:ntimestep){
        ntsparam <- c(alpha, theta, B_params['betas' ,t], B_params['gammas' ,t])
        u_i <- u[i, t]
        r <- ipnts(u_i, ntsparam)
        paths[i, t] <- r
      }
    }
  }
  if(parallel == TRUE){
    stopCluster(cluster)
  }
  return(paths)
}

#################################################################################################################################################################################

#################################################################################################################################################################################
#' @description Generates a series of GARCH 1, 1 sigma paths with errors assumed to follow a stdNTS distribution with time varying beta and gamma parameters.
#' @param sample_errors The result of gensamplestoTntserrors. Returns a matrix of the same dimensions npath x ntimestep.
#' @param kappa 
#' @param xi
#' @param lambda
#' @param zeta
#' @param sigma0 The initial value of sigma.
#' @return A npath x ntimestep matrix of sigma paths following the GARCH 1, 1 process with errors assumed to be stdNTS with stochastic time varying parameters beta and gamma.
## TODO: Assumes constant lambda, TODO: Make lambda dynamic
## TODO: Make lookbacks dynamic, not just GARCH 1, 1
gensamplesigmapaths <- function(sample_errors, kappa, xi, lambda, zeta, sigma0){
  paths <- matrix(0, nrow = nrow(sample_errors), ncol = ncol(sample_errors))
  for(i in 1:nrow(paths)){
    paths[i, 1] <- sigma0
    for(t in 2:ncol(paths)){
      sigma_today <- paths[i, t-1]
      error_today <- sample_errors[i, t-1]
      sigma_tomorrow <- sqrt(kappa + xi * sigma_today^2 * (error_today - lambda)^2 + zeta * sigma_today^2)
      paths[i, t] <- sigma_tomorrow
    }
  }
  return(paths)
}
#################################################################################################################################################################################
#' @description Generates series of risk neutral asset prices following the stoT-NTS process.
#' @param alpha The alpha parameter of the stdNTS distribution. Assumed to be a static number between 0 and 2, not inclusive.
#' @param theta The theta parameter of the stdNTS distribution. Assumed to be a static number above 0.
#' @param npath The number of paths to simulate
#' @param ntimestep The number of timesteps to simulate.
#' @param dt 
#' @param B_params The result of the gensampleBpath function with calculate_beta_gamma = TRUE. Must have matching npath and ntimestep.
#' @param r Rate of risk free return. Currently, only a static number is supported.
#' @param d Dividend. Currently, only a flat number is supported.
#' @param sigma 
#' @param error
#' @param S0 Initial asset price
#' @param y0 Initial asset return
#' @return An npath x ntimestep matrix of risk neutral asset prices following the stoT-NTS process
# Assume r constant, d = 0. ## TODO: Change later
genrfsamplestoTNTSprices <- function(alpha, theta, B_params, kappa, xi, lambda, zeta, sigma0, S0, y0 = 0, npath, ntimestep, dt = 1/250, r = 1/250, d=0){
  ## TODO:: Add in B for beta
  error <- gensamplestoTntserrors(alpha, theta, npath, ntimestep, dt, B_params, parallel = FALSE)
  sigma <- gensamplesigmapaths(error, kappa, xi, lambda, zeta, sigma0)
  paths <- matrix(0, nrow = nrow(error), ncol = ncol(error))
  for(i in 1:nrow(paths)){
    paths[i, 1] <- y0
    for(t in 2:ncol(paths)){
      if(!is.null(B_params)){
        w = log(chf_stdNTS(u = -1i * sigma[i, t], param = c(alpha, theta, B_params['betas',t], B_params['gammas' ,t])))
        paths[i, t] <- r - d - w + sigma[i, t] * error[i, t]
      } else {
        w = log(chf_stdNTS(u = -1i * sigma[i, t], param = c(alpha, theta, B_params['betas',t], B_params['gammas' ,t])))
        paths[i, t] <- r - d - w + sigma[i, t] * error[i, t]
      }
      
    }
  }
  paths <- t(apply(paths, MARGIN = 1, FUN = cumsum))
  prices <- S0 * exp(paths)
  # Convert back to real number. chf_stdNTS(u=-1i) gives all complex as +0i
  prices <- Re(prices)
  return(prices)
}

#################################################################################################################################################################################

#################################################################################################################################################################################
#' @description Generates series of risk neutral asset prices following the stoT-NTS process. Computes them in parallel
#' @param alpha The alpha parameter of the stdNTS distribution. Assumed to be a static number between 0 and 2, not inclusive.
#' @param theta The theta parameter of the stdNTS distribution. Assumed to be a static number above 0.
#' @param npath The number of paths to simulate
#' @param ntimestep The number of timesteps to simulate.
#' @param dt Time unit. Default 1/250 i.e one day.
#' @param B_params The result of the gensampleBpath function with calculate_beta_gamma = TRUE. Only one path's params must be supplied. Must have matching npath and ntimestep.
#' @param r Rate of risk free return. Currently, only a static number is supported.
#' @param d Dividend. Currently, only a flat number is supported.
#' @param sigma 
#' @param error
#' @param S0 Initial asset price
#' @param y0 Initial asset return
# Assume r constant, d = 0. ## TODO: Change later
library(parallel)
library(doParallel)
library(foreach)
library(temStaR)
genrfsamplestoTNTSprices_parallel <- function(alpha, theta, B_params, kappa, xi, lambda, zeta, sigma0, S0 = 100, y0 = 0, npath, ntimestep, dt = 1/250, r = 1/250, d=0, ncores = detectCores() - 1) {
  
  if(is.list(B_params)){
    stop("Only index one path's B_params for a single call of this function. Use lapply to do so for all paths.")
  }
  cl <- makeCluster(ncores)
  error <- gensamplestoTntserrors(alpha, theta, npath, ntimestep, dt, B_params, parallel = TRUE, cluster = cl)
  sigma <- gensamplesigmapaths(error, kappa, xi, lambda, zeta, sigma0)
  cl <- makeCluster(ncores)
  registerDoParallel(cl) # TODO: Fix this gensamplestoTntserrors should try to access cluster if needed instead.

  paths_list <- foreach(i = 1:nrow(error), .combine = rbind, .packages = c("temStaR")) %dopar% {
    path <- numeric(ncol(error))
    path[1] <- y0
    for (t in 2:ncol(error)) {
      w <- log(chf_stdNTS(u = -1i * sigma[i, t], param = c(alpha, theta, B_params['betas', t], B_params['gammas', t])))
      path[t] <- r - d - w + sigma[i, t] * error[i, t]
    }
    cumsum(path)  
  }

  stopCluster(cl)

  prices <- S0 * exp(paths_list)
  # Convert back to real number. chf_stdNTS(u=-1i) gives all complex as +0i
  prices <- Re(prices)

  return(prices)
}
#################################################################################################################################################################################
#################################################################################################################################################################################
#' @description Transforms sample stoT-NTS risk neutral price paths into a risk neutral option price path or several price paths.
#' @param r Risk free rate of return. Currently only supports static number
#' @param pct_otm A vector of moneyness values to price the put(s) and call(s)
#' @param t  The time until expiry. 
#' @param sample_prices The result of the gensamplestoTntsprices() function call
#' @param paths If single, will only return one set of option price paths for a put and call created at the initial value. If many, will generate a separate path for each day.
#' @return For paths = 'single', a list of two length(pct_otm) x ntimestep matrices representing a call and put price path, respectively. If paths = 'many', 
gensamplerfoptionprices <- function(r = .02/250, moneyness = 1.5, t = 90, sample_prices, type = c('US', 'European')) {

  # Define the maximum starting time for a t day option. 
  t <- ncol(sample_prices)
  # If we are only interested in a single option path, will only compute a path for one option. Otherwise, calculates a new one every single day,.
  discount_factors <- exp(-r * (0:(t - 1)))
  
  # these whill be lists of matrices storing the option prices at different starting points.
  call_matrix <- matrix(NA, nrow = 1, ncol = t)
  put_matrix  <- matrix(NA, nrow = 1, ncol = t)
  
  
  strike <- sample_prices[, 1] * moneyness
  
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
  

  
  if(type == "European"){
    call_price <- call_matrix[, ncol(call_matrix)]
    put_price <- put_matrix[, ncol(put_matrix)]
    option_prices <- c(call_price, put_price)
    names(option_prices) <- c('Call', 'Put')
  } else if(type == "US"){
    call_price <- call_matrix
    put_price <- put_matrix
    option_prices <- list(call_matrix, put_matrix)
    names(option_prices) <- c("Call", "Put")
  }
  
  return(option_prices)

 
}

#################################################################################################################################################################################

# I think its justified to have B_t as an input rather than the parameters used to make B_t. Why?
# - B_t is effectively just a sum of noise. We would need TONS of simulations in order to capture the average value of B_t given some parameters of B.. mind you that also isnt useful in production. People want to know what their parameters are, not the metaparameters and then estimate it from those. 
# Another thing to keep in mind is that B_t is our ONLY stochastic parameter. Maybe we should use an embedded network that aims to estimate B_t directly, without the influence of other parameters? This might help save on training time.
#################################################################################################################################################################################
#' @description Wrapper function capable of producing option prices en masse alongside vectors of the parameters used to create them
#' @return A ntimestep x parameters + 1 dimension matrix. The option prices are the last column, the parameters used to produce the ith option price are all other columns.
stoTntsoption <- function(nBpath, npath, alpha, theta, a0, a1, sigma_error, B0, kappa, xi, lambda, zeta, sigma0, S0, y0, moneyness, tao, r = 0.2/250, ncores = detectCores() - 1) {
  ntimestep <- ceiling(tao * 250)
  
  # Generate B paths with beta and gamma calculations
  B_paths <- gensampleBpath(alpha, theta, nBpath, ntimestep, a0, a1, sigma_error, B0 = B0, calculate_beta_gamma = TRUE)
  
  # Generate risk-neutral asset price paths
  sample_prices <- lapply(B_paths, function(B_param) {
    genrfsamplestoTNTSprices_parallel(
      alpha = alpha, theta = theta, B_params = B_param, kappa = kappa, xi = xi, 
      lambda = lambda, zeta = zeta, sigma0 = sigma0, S0 = S0, y0 = y0, 
      npath = npath, ntimestep = ntimestep, dt = 1/250, r = r, ncores = ncores
    )
  })
  # Generate option prices
  option_prices <- lapply(sample_prices, function(price_matrix) {
    gensamplerfoptionprices(
      r = r, moneyness = moneyness, sample_prices = price_matrix, type = 'European'
    )
  })
  # Combine call and put prices into a matrix
  result_df <- as.data.frame(do.call(rbind, option_prices))
  # Convert to values
  result_df <- result_df / S0
  return(result_df)
}

######################################### Monte Carlo Simulation Workflow #########################################
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
  halton_points[, 8] <- halton_points[, 8] * (1 - xi)
  
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
  return(halton_points)
}
