library(readr)
library(plyr)
library(dplyr)
library(reshape2)
library(parallel)
library(caret)
library(xgboost)
library(doMC)
library(pROC)
usetopvars <- function(df) {
  topvars <- c('disposition',
               'esi',
               'age', 'gender', 'maritalstatus', 'employstatus','insurance_status', 
               'race','ethnicity','lang','religion',
               'n_edvisits', 'previousdispo', 'n_admissions', 'n_surgeries',
               names(df)[which(names(df) == 'meds_analgesicandantihistaminecombination'):which(names(df) == 'meds_vitamins')])
  df[,names(df) %in% topvars]
}

topvar_df <- usetopvars(df)

rm(df)

makematrix <- function(df, sparse = T) {
  library(Matrix)
  df$disposition <- as.numeric(df$disposition == 'Admit')
  response <- df$disposition
  df <- select(df,-disposition)
  
  #dummify categorical variables and encode into matrix
  dmy <- dummyVars(" ~ .", data = df)
  if (sparse) {
    df <- Matrix(predict(dmy, newdata = df), sparse = T)
  } else {
    df <- predict(dmy, newdata = df)
  }
  
  list(y = response, x = df)
}

data_sparse <- makematrix(topvar_df, sparse = T) 

splitdataindex <- function(df, n = 5) {
  set.seed(3883)
  indeces_list <- vector("list", n)
  i_all <- as.numeric(rownames(df))
  i_test <- sample(i_all, 56000) 
  i_traindev <- setdiff(i_all, i_test)
  for (i in 1:n) {
    i_dev <- sample(i_traindev, 56000)
    i_train <- setdiff(i_traindev, i_dev)
    indeces_list[[i]] <- list(i_train = i_train, i_dev = i_dev, i_test = i_test)
  }
  indeces_list
}
indeces_list <- splitdataindex(topvar_df) 


fitboost <- function(dataset, indeces, 
                     max_depth, 
                     eta,
                     nthread,
                     nrounds,
                     colsample_bylevel) {
  x <- dataset$x
  y <- dataset$y
  rm(dataset)
  
  x_train <- x[indeces$i_train,]
  y_train <- y[indeces$i_train]
  
  x_dev <- x[indeces$i_dev,]
  y_dev <- y[indeces$i_dev]
  
  x_test <- x[indeces$i_test,]
  y_test <- y[indeces$i_test]
  
  rm(x); rm(y)
  
  bst <- xgboost(data = x_train, label = y_train,
                 max_depth = max_depth, eta = eta,
                 nthread = nthread, nrounds = nrounds,
                 eval_metric = 'auc',
                 objective = "binary:logistic",
                 colsample_bylevel = colsample_bylevel)
  print(bst)
  auc_train <- as.numeric(bst$evaluation_log$train_auc[length(bst$evaluation_log$train_auc)])
  
  #7) Predict on dev
  y_hat_dev <- predict(bst, x_dev)
  auc_dev <- as.numeric(auc(y_dev, y_hat_dev))
  
  c(auc_train,auc_dev)
}

results <- matrix(NA, length(indeces_list), 2)
colnames(results) <- c('train', 'dev')

for (depth in c(15,20,25)) {
  for (i in 1:length(indeces_list)) {
    indeces <- indeces_list[[i]]
    aucs <- fitboost(data_sparse, indeces, 
                     max_depth = depth, 
                     eta = 0.3,
                     nthread = 5,
                     nrounds = 30,
                     colsample_bylevel = 0.05)
    results[i,] <- aucs
  }
  print(results)
  print(paste('Average train and dev AUCs for depth', depth))
  print(colMeans(results))
  
}

indeces_bayesOp <- indeces_list[[1]]             

x <- data_sparse$x
y <- data_sparse$y
x_train <- x[-indeces_bayesOp$i_test,]
y_train <- y[-indeces_bayesOp$i_test]

x_test <- x[indeces_bayesOp$i_test,]
y_test <- y[indeces_bayesOp$i_test]

rm(x); rm(y)

bayes_opt <- function(dataset) {
  res0 <- xgb_cv_opt(data = x_train,
                     label = y_train,
                     objectfun = "binary:logistic",
                     evalmetric = "auc",
                     n_folds = 5,
                     acq = "ucb",
                     init_points = 10,
                     n_iter = 20)
  
  res0
}    

bayes_hyperPs = bayes_opt(data_sparse)

best_bayesOp <- xgboost(data = x_train, label = y_train,
                        max_depth = 5, eta = 0.3241,
                        nthread = 5, nrounds = 138.7780,
                        eval_metric = 'auc',
                        objective = "binary:logistic",
                        colsample_bylevel = 0.8150)

y_hat_bayes <- predict(best_bayesOp, x_test)
auc_test <- as.numeric(auc(y_test, y_hat_bayes))

best_nobayesOp <- xgboost(data = x_train, label = y_train,
                          max_depth = 10, eta = 0.3,
                          nthread = 5, nrounds = 20,
                          eval_metric = 'auc',
                          objective = "binary:logistic",
                          colsample_bylevel = 0.5)

y_hat_nobayesOp <- predict(best_nobayesOp, x_test)
auc_test_nobayes <- as.numeric(auc(y_test, y_hat_nobayesOp))