library(forecTheta)
library(mltools)
library(xgboost)
library(forecast)
library(caret)
library(e1071)
library(data.table)
library(lubridate)
library(stringr)
library(imputeTS)

#' Ranks seven forecasting methods for a time series or data frame object
#'
#' This is a function to forecast based on the the seven methods: random walk,
#' arima, exponential smoothing, xgboost, svm, ann and cart for a given 
#' time series or data frame. Then based on the error measures mape and mdape
#' are the results compared and at ranking of the methods is created and 
#' returned. For a data.frame object it takes the first column as forecasting
#' attribute and the rest as features. As forecasting horizon the last 10%
#' observations of the input data are taken. As input is only required an object 
#' from the class data.frame or time series.The data.frame object should 
#' represent an multivariate ts. Otherwise the function returns an error message.
#' 
#' @param eval_data A time series or data frame object.
#' @return A vector containing containing descending the method ranking. 
#' If the input is not a time series or data frame object, an error message
#' is returned.
#' @examples
#' forecast_evaluation(eval_data = datasets::BJsales)
#' forecast_evaluation(eval_data = datasets::AirPassengers)
forecast_evaluation <- function(eval_data){
  
  if (is.ts(eval_data) | is.data.frame(eval_data)) {
    
    # Prepare the data, na imputation and transform all to numeric
    eval_data <- forecast_data_preparation(eval_data = eval_data)
    
    # Set the horizon (max. 30 days)
    fs_horizon <- round(nrow(as.data.frame(eval_data)) * 0.10)
    if (fs_horizon > 30) {
      fs_horizon <- 30
    }
    
    print(fs_horizon)
    # Get the actual data
    actual_data <- as.numeric(eval_data[(1+(nrow(eval_data)-
                                              fs_horizon)):nrow(eval_data), "label"])
    # 0 values have to be change because the MAPE cannot handle 0s
    actual_data[which(actual_data == 0)] <- 0.0001
    # Generate all forecasts for the seven defined methods
    # Get the rw forecast
    eval_rw <- forecast_rw(ts = eval_data[, "label"]
                           , horizon = fs_horizon)
    # Get the arima forecast
    eval_arima <- forecast_arima(ts = eval_data[, "label"]
                                 , horizon = fs_horizon)
    # Get the es forecast
    eval_es <- forecast_es(ts = eval_data[, "label"]
                           , horizon = fs_horizon)
    

    best_mape <- 1000000000
    # Find the best svm forecast for the three possible methods
    for(method in c('svmPoly', 'svmLinear')){
      tmp_svm <- forecast_svm(fs_data = eval_data, svm_method = method, horizon = fs_horizon
                              , data_label = "label", data_features = 
                                colnames(eval_data)[which(colnames(eval_data) != "label")])
      # 0 values have to be change because the MAPE cannot handle 0s
      tmp_svm = as.numeric(tmp_svm)
      tmp_svm[which(tmp_svm == 0)] <- 0.0001
      tmp_mape <- forecTheta::errorMetric(obs = actual_data,
                                          forec = as.numeric(tmp_svm),
                                          type="APE", statistic="M")
      print(tmp_mape)
      # Get the forecast of the method with the smallest mape
      if (tmp_mape < best_mape) {
        # eval_svm contains the best forecast
        eval_svm <- tmp_svm
        best_mape <- tmp_mape
      }
    }
    
    best_mape <- 1000000000
    # Find the best xgboost forecast for the three possible methods
    for(method in c('xgbTree', 'xgbLinear')){
      tmp_xgb <- forecast_xgboost(fs_data = eval_data, xgb_method = method, horizon = fs_horizon
                                  , data_label = "label", data_features = 
                                    colnames(eval_data)[which(colnames(eval_data) != "label")])
      
      # 0 values have to be change because the MAPE cannot handle 0s
      tmp_xgb = as.numeric(tmp_xgb)
      tmp_xgb[which(tmp_xgb == 0)] <- 0.0001
      
      tmp_mape <- forecTheta::errorMetric(obs = actual_data,
                                          forec = as.numeric(tmp_svm),
                                          type="APE", statistic="M")
      # Get the forecast of the method with the smallest mape
      if (tmp_mape < best_mape) {
        # eval_xgb contains the best forecast
        eval_xgb <- tmp_xgb
        best_mape <- tmp_mape
      }
    }
    
    # Get the ann forecast
    eval_ann <- forecast_ann(fs_data = eval_data, horizon = fs_horizon
                             , data_label = "label", data_features = 
                               colnames(eval_data)[which(colnames(eval_data) != "label")])
    
    best_mape <- 1000000000
    # Find the best cart forecast for the three possible methods
    for(method in c('rpart')){
      tmp_cart <- forecast_cart(fs_data = eval_data, cart_method = method, horizon = fs_horizon
                                , data_label = "label", data_features = 
                                  colnames(eval_data)[which(colnames(eval_data) != "label")])
      # 0 values have to be change because the MAPE cannot handle 0s
      tmp_cart = as.numeric(tmp_cart)
      tmp_cart[which(tmp_cart == 0)] <- 0.0001
      
      tmp_mape <- forecTheta::errorMetric(obs = actual_data,
                                          forec = as.numeric(tmp_svm),
                                          type="APE", statistic="M")
      
      # Get the forecast of the method with the smallest mape
      if (tmp_mape < best_mape) {
        # eval_cart contains the best forecast
        eval_cart <- tmp_cart
        best_mape <- tmp_mape
      }
    }
    
  } else {
    stop("A time series or data frame object is required as input!")
  }
  
  # Generate the MAPE and MdAPE for the forecasting results
  mape_vector <- c()
  mdape_vector <- c()
  for(elem in list(eval_rw, eval_arima, eval_es, eval_svm,
                   eval_xgb, eval_ann, eval_cart)){
    calc_mape <- forecTheta::errorMetric(obs = actual_data,
                                         forec = as.numeric(elem), type="APE", statistic="M")
    mape_vector <- append(mape_vector, calc_mape)
    calc_mdape <- forecTheta::errorMetric(obs = actual_data,
                                          forec = as.numeric(elem), type="APE", statistic="Md")
    mdape_vector <- append(mdape_vector, calc_mdape)
  }
  # Assigning the method names to the above generated values
  methods <- c("rw", "arima", "es", "svm", "xgb", "ann", "cart")
  names(mape_vector) <- methods
  names(mdape_vector) <- methods
  
  # Sort and weight the evaluation results
  sort_mape <- c(7,6,5,4,3,2,1)
  sort_mdape <- c(7,6,5,4,3,2,1)
  names(sort_mape) <- names(sort(mape_vector, decreasing = FALSE))
  names(sort_mdape) <- names(sort(mdape_vector, decreasing = FALSE))
  
  # Combine the mape and mdape results
  final_evaluation = c()
  for(elem in methods){
    combine_errors = sort_mape[elem] + sort_mdape[elem]
    final_evaluation <- append(final_evaluation, combine_errors)
  }
  
  return(names(sort(final_evaluation, decreasing = TRUE)))
  
}




#' Prepares a time series or data frame object for the forecasting
#'
#' This is a function to prepares a time series or data frame object for the
#' forecasting. As na imputation methods are provided: 'mean' and 'kalman'.
#' Furthermore, all date cols are deleted. Also, all cols are transformed into
#' numeric values. As input is only required an objectfrom the class data.frame
#' or time series.The data.frame object should represent an multivariate ts.
#' Otherwise the function returns an error message.
#'
#' @param eval_data A time series or data frame object.
#' @param na_option A string value containing either 'mean'
#' or'kalman'; Default value is 'kalman'.
#' @return A data frame object containg the prepared data. 
#' If the input is not a time series or data frame object, an error message
#' is returned.
#' @examples
#' forecast_data_preparation(eval_data = datasets::BJsales, na_option = 'kalman')
#' forecast_data_preparation(eval_data = datasets::AirPassengers)
forecast_data_preparation <- function(eval_data, na_option = "mean"){
  
  if (is.ts(eval_data)) {
    eval_data = data.frame(label = as.numeric(eval_data),
                             feature = as.numeric(1:length(eval_data)))
    
  } else if (is.data.frame(eval_data)) {
    
    prep_eval_data <- eval_data
    
    # The date col has to be deleted
    date_index <- colnames(prep_eval_data)[which(str_detect(
      colnames(prep_eval_data),regex("date", ignore_case = TRUE)))]
    prep_eval_data[, date_index] <- NULL
    
    # Delete features that have only one or none unique value
    for(col in colnames(prep_eval_data)){
      if(length(unique(na.omit(prep_eval_data[, col]))) == 1 | 
         length(unique(na.omit(prep_eval_data[, col]))) == 0) {
        prep_eval_data[, col] <- NULL
      }
    }
    
    # All date cols have to be deleted
    for (colname in c(names(Filter(is.Date, prep_eval_data)),
                      names(Filter(is.POSIXt, prep_eval_data)))) {
      prep_eval_data[, colname] <- NULL
    }
    
    # First numeric feature is the forecasting label
    forecast_label <- names(Filter(is.numeric, prep_eval_data))[1]
    
    print(prep_eval_data[, forecast_label])
    
    # Preprocess the data without the forecast_label
    prep_eval_data <- prep_eval_data[, colnames(prep_eval_data)[which(
    colnames(prep_eval_data) != forecast_label)]]
    
    prep_eval_data <- as.data.frame(prep_eval_data)
    
    if(ncol(prep_eval_data) == 0) {
      prep_eval_data <- data.frame( feature = as.numeric(1:nrow(eval_data)))
    }
    
    # All character values have to be transformed as factor
    for (colname in names(Filter(is.character, prep_eval_data))) {
      prep_eval_data[, colname] <- as.factor(prep_eval_data[, colname])
    }
    
    # One-Hot-Encode unordered factor columns of a data.table
    encoded_data = mltools::one_hot(data.table::as.data.table(prep_eval_data)
                                    , cols = "auto", dropCols = TRUE, dropUnusedLevels = TRUE)
    
    
    # Combine preprocessed data with forecast_label
    eval_data <- cbind(as.data.frame(encoded_data), data.frame(label = eval_data[, forecast_label]))
    
  } else {
    stop("A time series or data frame object is required as input!")
    
  }
  
  
  
  # If else clause to check the 'na_option' input parameter
  if (na_option == "kalman") {
    for (col in colnames(eval_data)) {
      if (length(na.omit(eval_data[,col])) != length(eval_data[,col])) {
        if ((length(eval_data[,col]) - length(na.omit(as.numeric(eval_data[,col])))) < 3){
          if (length(na.omit(as.numeric(eval_data[,col]))) == 0) {
            eval_data[,col] <- NULL
          } else {
            eval_data[,col] <- na.mean(eval_data[,col], option = "mean")
            print("For kalman imputation are at least 3 NA observations
                  + required, thus the simple 'mean' imputation is used")
          }
          
          } else {
            # Replace all NA by na.kalman
            eval_data[,col] <- na.kalman(eval_data[,col])
          }
        
      }
    }
  } else if (na_option == "mean") {
    for (col in colnames(eval_data)) {
      if (length(na.omit(eval_data[,col])) != length(eval_data[,col])) {
        if (length(na.omit(as.numeric(eval_data[,col]))) == 0) {
          eval_data[,col] <- NULL
        } else {
          # Replace all NA by na.mean
          eval_data[,col] <- na.mean(eval_data[,col], option = "mean")
        }
      }
    }
    
  } else {
    stop("ERROR: input parameter 'na_option' has to contain either
         'mean' or 'kalman'")
  }
  
  
  return(eval_data)
  
  }





