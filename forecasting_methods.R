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

#' Forecast for a ts object based on random walk method
#'
#' This is a function to forecast based on the random walk method for a given
#' time series according to the inputed horizon.
#' As input is only required an object from the class time series.
#' Otherwise the function returns an error message.
#'
#' @param ts A time series object.
#' @param horizon The forecasting horizon for \code{ts}. The default value
#' is the last 10% observations of the \code{ts}.
#' @return The forecast values for the given \code{horizon}. If the input is 
#' not a ts, an error message is returned.
#' @examples
#' forecast_rw(ts = datasets::BJsales)
forecast_rw <- function(ts, horizon = round(length(ts) * 0.10)){
  # If else clause to check the input
  if (is.ts(ts) | is.vector(ts)) {
    rw_forecast = forecast::rwf(ts[1:(length(ts)-horizon)], h=horizon)
    return(as.numeric(rw_forecast$mean))
  } else {
    print("A time series object is required as input!")
  }
}

#' Forecast for a ts object based on auto arima method
#'
#' This is a function to forecast based on the auto arima method for a given
#' time series according to the inputed horizon. That means, based on the
#' inputed data is the best fitting arima model selected.
#' As input is only required an object from the class time series.
#' Otherwise the function returns an error message.
#'
#' @param ts A time series object.
#' @param horizon The forecasting horizon for \code{ts}. The default value
#' is the last 10% observations of the \code{ts}.
#' @return The forecast values for the given \code{horizon}. If the input is 
#' not a ts, an error message is returned.
#' @examples
#' forecast_arima(ts = datasets::BJsales)
forecast_arima <- function(ts, horizon = round(length(ts) * 0.10)){
  # If else clause to check the input
  if (is.ts(ts) | is.vector(ts)) {
    fitted_arima <- forecast::auto.arima(ts[1:(length(ts)-horizon)])
    arima_forecast <- forecast::forecast(fitted_arima, h = horizon)
    return(as.numeric(arima_forecast$mean))
  } else {
    print("A time series object is required as input!")
  }
}

#' Forecast for a ts object based on auto exponential smoothing method
#'
#' This is a function to forecast based on the auto exponential smoothing 
#' method for a given time series according to the inputed horizon. That 
#' means, based on the inputed data is the best fitting model selected.
#' As input is only required an object from the class time series.
#' Otherwise the function returns an error message.
#'
#' @param ts A time series object.
#' @param horizon The forecasting horizon for \code{ts}. The default value
#' is the last 10% observations of the \code{ts}.
#' @return The forecast values for the given \code{horizon}. If the input is 
#' not a ts, an error message is returned.
#' @examples
#' forecast_es(ts = datasets::BJsales)
forecast_es <- function(ts, horizon = round(length(ts) * 0.10)){
  # If else clause to check the input
  if (is.ts(ts) | is.vector(ts)) {
    fitted_ets <- forecast::ets(ts[1:(length(ts)-horizon)])
    ets_forecast <- forecast::forecast(fitted_ets, h = horizon)
    return(as.numeric(ets_forecast$mean))
  } else {
    print("A time series object is required as input!")
  }
}

#' Forecast for a ts or data frame object based on svm methods
#'
#' This is a function to forecast based on the svm methods for a given 
#' time series or data frame according to the inputed horizon. 
#' As input is only required an object from the class data.frame or time series.
#' The data.frame object should represent an multivariate ts.
#' Otherwise the function returns an error message.
#' Also, the three different methods \code{svm_method} are allowed:
#' 'svmPoly', 'svmRadial' or 'svmLinear'. Otherwise the function returns 
#' an error message.
#'
#' @param fs_data A time series or data frame object.
#' @param horizon The forecasting horizon for \code{fs_data}. The default value
#' is the last 10% observations of the \code{fs_data}.
#' @param n_round number of runs for the training model. Default is 10.
#' @param cv_nfold number of folds for cross validation. Default is 10.
#' @param data_features For a time series object it can be an empty vector 
#' (default). For a data frame it is a character vector with all col names of 
#' the data frame used as features for the forecast.
#' @param data_label For a time series object it can be an empty vector 
#' (default). For a data frame it is the col name for the attribute to forecast.
#' @param svm_method A svm method from the caret R package. Allowed are the
#' three strings: 'svmPoly', 'svmRadial' or 'svmLinear'. Default is the 
#' 'svmLinear'method.
#' @return The forecast values for the given \code{horizon}. If the input is 
#' not a time series or data frame object, an error message is returned.
#' @examples
#' forecast_svm(fs_data = datasets::BJsales)
#' forecast_svm(fs_data = datasets::AirPassengers, 
#' data_features = c("Feb","Mar","Apr"), data_label = "Jan")
forecast_svm <- function(fs_data ,horizon = round(nrow(as.data.frame(fs_data)) * 0.10)
                         , n_round = 10, cv_nfold = 10, data_features = c(),
                         data_label = c(), svm_method = "svmLinear"){
  
  # If else clause to check the input svm_method
  if (svm_method %in% c("svmPoly", "svmRadial", "svmLinear")) {
    
    # If else clause to check the input data
    if (is.ts(fs_data) | is.vector(fs_data)) {
      # Split data into training and testing data based on horizon
      train_data <- as.numeric(fs_data)[1:(length(fs_data)-horizon)]
      test_data <- as.numeric(fs_data)[(1+(length(fs_data)-horizon)):length(fs_data)]
      # Create training and testing dataframes
      train_df <- data.frame(label = as.numeric(train_data),
                             feature = as.numeric(1:length(train_data)))
      test_df <- data.frame(label = as.numeric(test_data),
                            feature <- as.numeric((length(train_data)+1):(length(train_data)
                                                                          +length(test_data))))
    } else if (is.data.frame(fs_data)) {
      # If else clause to check the input data_features and data_label
      if (length(data_features) > 0 & length(data_label) == 1) {
        # Split data into training and testing data based on horizon
        train_data <- fs_data[1:(nrow(fs_data)-horizon), c(data_label, data_features)]
        test_data <- fs_data[(1+(nrow(fs_data)-horizon)):nrow(fs_data),
                             c(data_label, data_features)]
        # Create training and testing dataframes
        train_df <- data.frame(label = train_data[, c(data_label)],
                               feature = train_data[, c(data_features)])
        test_df <- data.frame(label = test_data[, c(data_label)],
                              feature = test_data[, c(data_features)])
        
      } else {
        stop("Features of the train data are required!!!")
      }
      
    } else {
      stop("A time series or data frame object is required as input!")
    }
    
    # Fit the tuning parameters
    fit_control <- caret::trainControl(method = "repeatedcv",  
                                       number = cv_nfold,
                                       repeats = n_round,
                                       savePredictions = "final",
                                       summaryFunction = defaultSummary)
    
    # Train and identify a fitting model
    svm_model <- caret::train(label ~ ., data = train_df,                  
                              method = svm_method,
                              metric = 'RMSE',
                              trControl = fit_control, 
                              preProcess = c('scale', 'center'),
                              tuningLength = 5)
    # Forecast the final values
    svm <- predict(svm_model, newdata = test_df)
    return(svm)
    
  } else {
    stop("Only the three svm models: 'svmPoly', 'svmRadial', 'svmLinear' are allowed")
  }
  
}


#' Forecast for a ts or data frame object based on cart methods
#'
#' This is a function to forecast based on the cart methods for a given 
#' time series or data frame according to the inputed horizon. 
#' As input is only required an object from the class data.frame or time series.
#' The data.frame object should represent an multivariate ts.
#' Otherwise the function returns an error message.
#' Also, the three different methods \code{cart_method} are allowed:
#' 'rpart', 'rpart1SE', 'rpart2'. Otherwise the function returns 
#' an error message.
#'
#' @param fs_data A time series or data frame object.
#' @param horizon The forecasting horizon for \code{fs_data}. The default value
#' is the last 10% observations of the \code{fs_data}.
#' @param n_round number of runs for the training model.Default is 10.
#' @param cv_nfold number of folds for cross validation.Default is 10.
#' @param data_features For a time series object it can be an empty vector 
#' (default). For a data frame it is a character vector with all col names of 
#' the data frame used as features for the forecast.
#' @param data_label For a time series object it can be an empty vector 
#' (default). For a data frame it is the col name for the attribute to forecast.
#' @param cart_method A svm method from the caret R package. Allowed are the
#' three strings: 'rpart', 'rpart1SE' or 'rpart2'. Default is the 
#' 'rpart'method.
#' @return The forecast values for the given \code{horizon}. If the input is 
#' not a time series or data frame object, an error message is returned.
#' @examples
#' forecast_cart(fs_data = datasets::BJsales)
#' forecast_cart(fs_data = datasets::AirPassengers, 
#' data_features = c("Feb","Mar","Apr"), data_label = "Jan")
forecast_cart <- function(fs_data ,horizon = round(nrow(as.data.frame(fs_data)) * 0.10)
                          , n_round = 10, cv_nfold = 10, data_features = c(), 
                          data_label = c(), cart_method = "rpart"){
  
  # If else clause to check the input svm_method
  if (cart_method %in% c("rpart", "rpart1SE", "rpart2")) {
    
    # If else clause to check the input data
    if (is.ts(fs_data) | is.vector(fs_data)) {
      # Split data into training and testing data based on horizon
      train_data <- as.numeric(fs_data)[1:(length(fs_data)-horizon)]
      test_data <- as.numeric(fs_data)[(1+(length(fs_data)-horizon)):length(fs_data)]
      # Create training and testing dataframes
      train_df <- data.frame(label = as.numeric(train_data),
                             feature = as.numeric(1:length(train_data)))
      test_df <- data.frame(label = as.numeric(test_data),
                            feature <- as.numeric((length(train_data)+1):(length(train_data)+length(test_data))))
      
    } else if (is.data.frame(fs_data)) {
      # If else clause to check the input data_features and data_label
      if (length(data_features) > 0 & length(data_label) == 1) {
        # Split data into training and testing data based on horizon
        train_data <- fs_data[1:(nrow(fs_data)-horizon), c(data_label, data_features)]
        test_data <- fs_data[(1+(nrow(fs_data)-horizon)):nrow(fs_data), c(data_label, data_features)]
        # Create training and testing dataframes
        train_df <- data.frame(label = train_data[, c(data_label)],
                               feature = train_data[, c(data_features)])
        test_df <- data.frame(label = test_data[, c(data_label)],
                              feature = test_data[, c(data_features)])
        
      } else {
        stop("Features of the train data are required!!!")
      }
      
    } else {
      stop("A time series or data frame object is required as input!")
    }
    
    # Fit the tuning parameters
    fit_control <- caret::trainControl(method = "repeatedcv",  
                                       number = cv_nfold,
                                       repeats = n_round,
                                       savePredictions = "final",
                                       summaryFunction = defaultSummary)
    
    # Train and identify a fitting model
    cart_model <- caret::train(label ~ ., data = train_df, 
                               method = cart_method, 
                               trControl = fit_control,
                               ## Specify which metric to optimize
                               metric = "RMSE")
    # Forecast the final values
    cart <- predict(cart_model, newdata = test_df)
    return(cart)
    
  } else {
    stop("Only the three cart models: 'rpart', 'rpart1SE', 'rpart2' are allowed")
  }
  
}

#' Forecast for a ts or data frame object based on ann method
#'
#' This is a function to forecast based on the ann method for a given 
#' time series or data frame according to the inputed horizon. 
#' As input is only required an object from the class data.frame or time series.
#' The data.frame object should represent an multivariate ts.
#' Otherwise the function returns an error message.
#'
#' @param fs_data A time series or data frame object.
#' @param horizon The forecasting horizon for \code{fs_data}. The default value
#' is the last 10% observations of the \code{fs_data}.
#' @param n_round number of runs for the training model. Default is 10.
#' @param cv_nfold number of folds for cross validation. Default is 10.
#' @param data_features For a time series object it can be an empty vector 
#' (default). For a data frame it is a character vector with all col names of 
#' the data frame used as features for the forecast.
#' @param data_label For a time series object it can be an empty vector 
#' (default). For a data frame it is the col name for the attribute to forecast.
#' @return The forecast values for the given \code{horizon}. If the input is 
#' not a time series or data frame object, an error message is returned.
#' @examples
#' forecast_ann(fs_data = datasets::BJsales)
#' forecast_ann(fs_data = datasets::AirPassengers, 
#' data_features = c("Feb","Mar","Apr"), data_label = "Jan")
forecast_ann <- function(fs_data ,horizon = round(nrow(as.data.frame(fs_data)) * 0.10)
                         , n_round = 10, cv_nfold = 10, data_features = c(), 
                         data_label = c()){
  # If else clause to check the input data
  if (is.ts(fs_data) | is.vector(fs_data)) {
    # Split data into training and testing data based on horizon
    train_data <- as.numeric(fs_data)[1:(length(fs_data)-horizon)]
    test_data <- as.numeric(fs_data)[(1+(length(fs_data)-horizon)):length(fs_data)]
    # Create training and testing dataframes
    train_df <- data.frame(label = as.numeric(train_data),
                           feature = as.numeric(1:length(train_data)))
    test_df <- data.frame(label = as.numeric(test_data),
                          feature <- as.numeric((length(train_data)+1):(length(train_data)+length(test_data))))
    
  } else if (is.data.frame(fs_data)) {
    # If else clause to check the input data_features and data_label
    if (length(data_features) > 0 & length(data_label) == 1) {
      # Split data into training and testing data based on horizon
      train_data <- fs_data[1:(nrow(fs_data)-horizon), c(data_label, data_features)]
      test_data <- fs_data[(1+(nrow(fs_data)-horizon)):nrow(fs_data), c(data_label, data_features)]
      # Create training and testing dataframes
      train_df <- data.frame(label = train_data[, c(data_label)],
                             feature = train_data[, c(data_features)])
      test_df <- data.frame(label = test_data[, c(data_label)],
                            feature = test_data[, c(data_features)])
      
    } else {
      stop("Features of the train data are required!!!")
    }
    
  } else {
    stop("A time series or data frame object is required as input!")
  }
  
  # Fit the tuning parameters
  fit_control <- caret::trainControl(method = "repeatedcv",  
                                     number = cv_nfold,
                                     repeats = n_round,
                                     savePredictions = "final",
                                     summaryFunction = defaultSummary)
  
  # Train and identify a fitting model
  ann_model <- caret::train(label ~ ., data = train_df,                  
                            method = 'nnet',
                            trControl = fit_control,
                            linout = TRUE)
  # Forecast the final values
  ann <- predict(ann_model, newdata = test_df)
  return(ann)
}

#' Forecast for a ts or data frame object based on xgboost methods
#'
#' This is a function to forecast based on the xgboost methods for a given 
#' time series or data frame according to the inputed horizon. 
#' As input is only required an object from the class data.frame or time series.
#' The data.frame object should represent an multivariate ts.
#' Otherwise the function returns an error message.
#' Also, the three different methods \code{xgb_method} are allowed:
#' 'xgbTree', 'xgbLinear' or 'xgbDART'. Otherwise the function returns 
#' an error message.
#'
#' @param fs_data A time series or data frame object.
#' @param horizon The forecasting horizon for \code{fs_data}. The default value
#' is the last 10% observations of the \code{fs_data}.
#' @param n_round number of runs for the training model.Default is 10.
#' @param cv_nfold number of folds for cross validation.Default is 10.
#' @param data_features For a time series object it can be an empty vector 
#' (default). For a data frame it is a character vector with all col names of 
#' the data frame used as features for the forecast.
#' @param data_label For a time series object it can be an empty vector 
#' (default). For a data frame it is the col name for the attribute to forecast.
#' @param xgb_method A svm method from the caret R package. Allowed are the
#' three strings: 'xgbTree', 'xgbLinear' or 'xgbDART'. Default is the 
#' 'xgbTree'method.
#' @return The forecast values for the given \code{horizon}. If the input is 
#' not a time series or data frame object, an error message is returned.
#' @examples
#' forecast_xgboost(fs_data = datasets::BJsales)
#' forecast_xgboost(fs_data = datasets::AirPassengers, 
#' data_features = c("Feb","Mar","Apr"), data_label = "Jan")
forecast_xgboost <- function(fs_data ,horizon = round(nrow(as.data.frame(fs_data)) * 0.10)
                             , n_round = 10, cv_nfold = 10, data_features = c(),
                             data_label = c(), xgb_method = "xgbTree"){
  
  # If else clause to check the input svm_method
  if (xgb_method %in% c("xgbTree","xgbLinear","xgbDART")) {
    
    # If else clause to check the input data
    if (is.ts(fs_data) | is.vector(fs_data)) {
      # Split data into training and testing data based on horizon
      train_data <- as.numeric(fs_data)[1:(length(fs_data)-horizon)]
      test_data <- as.numeric(fs_data)[(1+(length(fs_data)-horizon)):length(fs_data)]
      # Create training and testing dataframes
      train_df <- data.frame(label = as.numeric(train_data),
                             feature = as.numeric(1:length(train_data)))
      test_df <- data.frame(label = as.numeric(test_data),
                            feature <- as.numeric((length(train_data)+1):(length(train_data)
                                                                          +length(test_data))))
      
    } else if (is.data.frame(fs_data)) {
      # If else clause to check the input data_features and data_label
      if (length(data_features) > 0 & length(data_label) == 1) {
        # Split data into training and testing data based on horizon
        train_data <- fs_data[1:(nrow(fs_data)-horizon), c(data_label, data_features)]
        test_data <- fs_data[(1+(nrow(fs_data)-horizon)):nrow(fs_data), c(data_label, data_features)]
        # Create training and testing dataframes
        train_df <- data.frame(label = train_data[, c(data_label)],
                               feature = train_data[, c(data_features)])
        test_df <- data.frame(label = test_data[, c(data_label)],
                              feature = test_data[, c(data_features)])
        
      } else {
        stop("Features of the train data are required!!!")
      }
      
    } else {
      stop("A time series or data frame object is required as input!")
    }
    
    # Fit the tuning parameters
    train_control <- caret::trainControl(
      method = "repeatedcv",  
      number = cv_nfold,
      repeats = n_round,
      savePredictions = "final",
      summaryFunction = defaultSummary
    )
    
    # Train and identify a fitting model
    xgb_base <- caret::train(label ~ ., data = train_df,
                             trControl = train_control,
                             method = xgb_method,
                             verbose = TRUE
    )
    # Forecast the final values
    xgb <- predict(xgb_base, newdata = test_df)
    return(xgb)
    
  } else {
    stop("Only the three cart models: 'xgbTree', 'xgbLinear', 'xgbDART' are allowed")
  }
}


