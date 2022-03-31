rm(list=ls())
#loading the data:
getwd()
setwd("C://Users//arbas//Desktop//VSE//2nd Semester//EY Class")
data<-read.csv("mortgage_sample.csv")
training_set <- train
test_set <- test
remove(test)
#-------------------------------------------------------------------------------
#Packages:
install.packages("gmodels")
library(gmodels)
library(tidyverse)
library(dplyr)
install.packages("ggthemes")
library(ggthemes)
library(ggplot2) 
library(readxl)
install.packages("rpart.plot")
library(rpart.plot)
library(pROC)
#--------------------------------------------------------------------------------
#Checking the data:

str(data) #622489 obs. of  24 variables:
any(is.na(data)) #TRUE-> we will delete na's
#NA's are due to the sample-> private, for which we will not conduct any analysis. Therefore, we can delete.

data1<-na.omit(data)
str((data1)) #560991 obs. of  24 variables after deleting NA's (which were due to ->sample->private)

#We will also drop the column sample
data1<-data1[-c(24)]
#Now we have 560991 obs. and 23 variables.

#Checking for data distribution:

#We are using Cross Table as it easily computes descriptive statistics for the given variables.

CrossTable(data1$default_time)
#   |         0 |         1 | 
#   |-----------|-----------|
#   |    547372 |     13619 | 
#   |     0.976 |     0.024 | 
#   |-----------|-----------|

CrossTable(data1$FICO_orig_time)
CrossTable(data1$LTV_orig_time)

CrossTable(data1$status_time)

#    |         0 |         1 |         2 | 
#    |-----------|-----------|-----------|
#    |    523432 |     13619 |     23940 | 
#    |     0.933 |     0.024 |     0.043 | 
#    |-----------|-----------|-----------|

#------DATA SPLITTING-----------------------------------------------------------
set.seed(567)

#Storing row numbers for the TRAINING SET.Index_train-> 2/3 of the original data

index_train<-sample(1:nrow(data1),2/3*nrow(data1))
training_set <- data1[index_train, ]

#Creating test set:

test_set <- data1[-index_train, ]

#-----LOGISTIC REGRESSION-------------------------------------------------------
#simple logistic model with 2 predictors

logistic_model<-glm(default_time~LTV_orig_time + FICO_orig_time,
                    family = "binomial", data = training_set)
summary(logistic_model)

#extended logistic model with 6 predictors
logistic_model2<-glm(default_time~LTV_orig_time + FICO_orig_time+investor_orig_time+balance_orig_time+ Interest_Rate_orig_time + hpi_orig_time,
                    family = "binomial", data = training_set)
summary(logistic_model2)

#-----PREDICTING PROBABILITY----------------------------------------------------
predict_logistic<-predict(logistic_model, newdata = test_set, type = "response")
predict_logistic2<-predict(logistic_model2, newdata = test_set, type = "response")

#The range of the probability predictions
range(predict_logistic)

#0.007207138 0.193217299-> range seems to be large, which is a good indicator
#If the range is small it means that test_set cases don't lie far apart, 
#therefore the model might not be good in discriminating good and bad loans

#----EVALUATING THE RESULT OF LRM-----------------------------------------------

# The cut-off is basically a measure of risk tolerance of the financial institution
#Cut-off set at 15%:
pred_cutoff_15 <- ifelse(predict_logistic > 0.15, 1, 0)

# A confusion matrix can be created afterwards to calculate Accuracy and compare cut-offs

conf_matrix_15<-table(test_set$default_time,pred_cutoff_15)
accuracy_15 <- sum(diag(conf_matrix_15)) / nrow(test_set)
accuracy_15
# The accuracy for the model is  0.9762991

#Cut-off set at 50%:
pred_cutoff_50 <- ifelse(predict_logistic > 0.50, 1, 0)
conf_matrix_50<-table(test_set$default_time,pred_cutoff_50)
accuracy_50 <- sum(diag(conf_matrix_50)) / nrow(test_set)
accuracy_50
table(test_set$default_time,pred_cutoff_50)
#The accuracy for the model is 0.9763098

# Increasing cut-off from 15% to 50% slightly increases overall Accuracy of the model.

# *An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a 
#classification model at all classification thresholds*

#ROC Curve:
library("pROC")
roc_logistic<-roc(test_set$default_time,predict_logistic)
roc_logistic2<-roc(test_set$default_time,predict_logistic2)

plot(roc_logistic, col="purple")

#Area under the curve:
auc(roc_logistic) #0.6048
auc(roc_logistic2) #0.6638

#-------DECISION TREE-----------------------------------------------------------
library(rpart)  #for construction of decision tree

tree_model<-rpart(default_time~LTV_orig_time + FICO_orig_time,
             data = training_set, method = "class", parms = list(prior = c(0.60, 0.40)),
             control = rpart.control(cp = 0.001))

plot(tree_model, uniform = T)
text(tree_model)

#------PRUNING THE DECISION TREE--- --------------------------------------------
install.packages("rpart.plot")
library(rpart.plot)

#Pruning is needed in order to prevent over fitting, which can lead to inaccurate predictions
#Visualizing cross-validated error (X-val Relative Error) in relation 
# to the complexity parameter for tree_model:

plotcp(tree_model)

# *CP-> complexity parameter=>minimum improvement in the model needed at each node*
#Info table about CP, splits and errors. We should identify 
# which split has the minimum cross-validated error in tree.model

printcp(tree_model)
index_min<-which.min(tree_model$cptable[, "xerror"])
tree_min<-tree_model$cptable[index_min, "CP"]
prune_tree<-prune(tree_model, cp = tree_min)
prp(prune_tree)

#-----EVALUATING THE DECISION TREE ---------------------------------------------

pred_tree<-predict(tree_model,newdata = test_set)[ ,2]
pred_pruned_tree<-predict(prune_tree,newdata = test_set)[ ,2]

conf_matrix_tree<-table(test_set$default_time, pred_tree)
accuracy_tree<-sum(diag(conf_matrix_tree))/nrow(test_set)
accuracy_tree  #0.008074996

#ROC Curve for the two decision trees:

roc_tree<-roc(test_set$default_time,pred_tree)
roc_prune_tree<-roc(test_set$default_time,pred_pruned_tree)

plot(roc_tree, col = "black") 
lines(roc_prune_tree, col = "blue") 

#Area under the curve:
auc(roc_tree) #0.5943
auc(roc_prune_tree) #0.5943

## Area under the curve for Logistic regression is 0.6048, whereas for decision tree is 0.59.

#-------MACHINE LEARNING--------------------------------------------------------
#xgboost

#Please not that his is the basic model. 
#The detailed ML models and models with modification are available in Python script, 
#provided seperately.

install.packages("xgboost")
library("xgboost")
y = training_set$default_time
xgb <- xgboost(data = as.matrix(training_set), 
               label = y, 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 12,
               nthread = 3
)
# predict values in test set

xgb_pred <- predict(xgb, as.matrix(test_set))

auc(xgb_pred)

#-------RANDOM FOREST--------------------------------------------------------
install.packages("randomForest")
library("randomForest")

training_set$default_time <- as.factor(training_set$default_time)

rf <- randomForest(default_time ~ LTV_orig_time + FICO_orig_time, data = training_set, proximity=TRUE, ntree=200) 

print(rf)
pred_rf = predict(rf, newdata=test_set)
roc_rf<-roc(test_set$default_time,pred_rf)
auc(roc_rf) 



#-------SCORECARD---------------------------------------------------------------
install.packages("scorecard")
library("scorecard")

#bins based on logistic model 2:
bins <- woebin(training_set, y = "default_time", x = c("LTV_orig_time", "FICO_orig_time",
                                                     "investor_orig_time","balance_orig_time","Interest_Rate_orig_time",
                                                     "hpi_orig_time"))

#scorecard based on logistic model with 2 predictors
score_card<-scorecard(bins, logistic_model2, points0 = 600, odds0 = 1/19, pdo = 50,
          basepoints_eq0 = FALSE, digits = 0)
score_card


