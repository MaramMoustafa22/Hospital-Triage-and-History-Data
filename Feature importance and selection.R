library(readr)
library(dplyr)
library(reshape2)
library(parallel)
library(caret)
library(xgboost)
library(doMC)
library(pROC)
library(keras)

df <- clean_names(df)
#disposition
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

splitdataindex_testplateau <- function(df) {
  set.seed(3883)
  cuts <- c(0.01, 0.1, 0.3, 0.5, 0.8, 1)
  indeces_list <- vector("list", length(cuts))
  i_all <- as.numeric(rownames(df))
  i_test <- sample(i_all, 56000)
  i_train_all <- setdiff(i_all, i_test)
  for (i in 1:length(cuts)) {
    i_train <- sample(i_train_all, floor(length(i_train_all)*cuts[i]))
    indeces_list[[i]] <- list(i_train = i_train, i_test = i_test)
  }
  indeces_list
}

indeces_list <- splitdataindex_testplateau(df)
indeces <- indeces_list[[1]]

indeces <- indeces_list[[1]]
x <- dataset$x
y <- dataset$y
rm(dataset)

x_test <- x[indeces$i_test,]
y_test <- y[indeces$i_test]

x_train <- x[-indeces$i_test,]
y_train <- y[-indeces$i_test]


rm(x); rm(y)

#run XGBoost 100 times

for (i in 1:100) {
  bst <- xgboost(data = x_train, label = y_train,
                 max_depth = 20, eta = 0.3,
                 nthread = 5, nrounds = 30,
                 eval_metric = 'auc',
                 objective = "binary:logistic",
                 colsample_bylevel = 0.05)
  # get importance table
  importance <- xgb.importance(feature_names = x_train@Dimnames[[2]], model = bst)
  #extract gain
  importance <- importance[,c(1,2)]
  #change name of column
  label <- paste0("Gain", i)
  names(importance)[2] <- label
  if (i == 1) {
    result <- importance
  } else {
    result <- left_join(result, importance, by = 'Feature')
  }
  print(paste("Finished iteration",i))
}

bst_topvars <- xgboost(data = x_train, label = y_train,
                       max_depth = 10, eta = 0.3,
                       nthread = 5, nrounds = 20,
                       eval_metric = 'auc',
                       objective = "binary:logistic",
                       colsample_bylevel = 0.5)

bst_pred_test_topvars <- predict(bst_topvars, x_test)
roc(y_test, bst_pred_test_topvars)
ci.auc(roc(y_test, bst_pred_test_topvars), conf.level = 0.95)