########################### Disease Prediction Using ML ################################
setwd("D:/Education/My Courses/ACADGILD/Online/Project 2-Cancer")
cancer <- read.csv("CancerData.csv")

## See the structure of data
str(cancer)
summary(cancer)

# delete X variable from the data
cancer <- cancer[,!names(cancer) %in% c("X")]

## Dependent Variable - diagnosis
prop.table(table(cancer$diagnosis))

## 3 types of variable #1. Mean #2. SE #3. Worst
library(corrplot)
#### Mean
MeanVars = c(
  "fractal_dimension_mean",
  "symmetry_mean",
  "concave.points_mean", 
  "concavity_mean",
  "compactness_mean",
  "smoothness_mean", 
  "area_mean",
  "perimeter_mean",
  "texture_mean" ,
  "radius_mean"
)
library(corrplot)
corrplot(cor(cancer[,names(cancer) %in% MeanVars]), order = "hclust", tl.cex = 1, addrect = 8)

#### SE
SEVars = c(
  "fractal_dimension_se",
  "symmetry_se",
  "concave.points_se", 
  "concavity_se",
  "compactness_se",
  "smoothness_se", 
  "area_se",
  "perimeter_se",
  "texture_se" ,
  "radius_se"
)
corrplot(cor(cancer[,names(cancer) %in% SEVars]), order = "hclust", tl.cex = 1, addrect = 8)

#### Worst Variable
worstVars = c(
  "fractal_dimension_worst",
  "symmetry_worst",
  "concave.points_worst", 
  "concavity_worst",
  "compactness_worst",
  "smoothness_worst", 
  "area_worst",
  "perimeter_worst",
  "texture_worst" ,
  "radius_worst"
)
corrplot(cor(cancer[,names(cancer) %in% worstVars]), order = "hclust", tl.cex = 1, addrect = 8)
### Correlation Plot 
corrplot(cor(cancer[,3:ncol(cancer)]), number.cex = .9, method = "square", 
         order = "hclust", type = "full", tl.cex=0.8, tl.col = "black")

############# Remove highly correlated Variables ##########
library(caret)
library(dplyr)
cancer2 <- cancer %>% select(-findCorrelation(cor(cancer %>% select(-id, -diagnosis)),
                                              cutoff = 0.9))


############# Create Machine Learning Models ##############
library(caret)
set.seed(1815)
df3 <- cbind(diagnosis = cancer$diagnosis, cancer2)
index <- createDataPartition(df3$diagnosis, times = 1, p = 0.8, list = FALSE)
training <- df3[index, ]
testing <-  df3[-index, ]
library(caret)
df_control <- trainControl(method="cv",
                           number = 10,
                           classProbs = TRUE)


############ Apply Machine Learning Models
###1. Logistic Regression Model 
logit <- train(diagnosis ~., data = training, method = "glm", 
                         metric = "ROC", preProcess = c("scale", "center"), 
                         trControl = df_control)

logit_pred <- predict(logit, testing)
cm_logit <- confusionMatrix(logit_pred, testing$diagnosis, positive = "M")
cm_logit

###2. Random Forest
tuneGrid <- expand.grid(.mtry = c(1: 10))
rf <- train(diagnosis ~., data = training,
                     method = "rf", 
                     metric = 'Accuracy',
                     tuneGrid = tuneGrid,
                     trControl = df_control)

rf_pred <- predict(rf, testing)
cm_rf <- confusionMatrix(rf_pred, testing$diagnosis, positive = "M")
cm_rf

# estimate variable importance
importance <- varImp(rf, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

#################### 3. XGBoost ########################################## 
#preparing matrix 
library(xgboost)
dtrain <- xgb.DMatrix(as.matrix(training[,-1]), label = as.numeric(training$diagnosis)-1) 
dtest <- xgb.DMatrix(as.matrix(testing[,-1]), label=as.numeric(testing$diagnosis)-1)

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
               subsample=1, colsample_bytree=1, metric = "auc")
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 30, 
                 nfold = 5, showsd = T, stratified = T, maximize = F)

bestIter = which.min(xgbcv$evaluation_log$test_error_mean)

xgb1 <- xgb.train (params = params, data = dtrain, nrounds = bestIter, 
                   watchlist = list(val=dtest,train=dtrain),eval_metric = "auc")
#model prediction
xgbpred <- predict(xgb1,dtest)
xgb_pred <- as.numeric(xgbpred > 0.5)
cm_xgboost <- confusionMatrix(xgb_pred, as.numeric(testing$diagnosis)-1)
cm_xgboost

# Variable Importance
mat <- xgb.importance(feature_names = colnames(dtrain),model = xgb1)
xgb.plot.importance (importance_matrix = mat, top_n = 15) 

############# Data Pre-Processing #########################
#### Apply PCA
library(dplyr)
pca_data <- prcomp(cancer %>% select(-id, -diagnosis), scale = TRUE, center = TRUE)
summary(pca_data)

# Calculate the proportion of variance explained
var <- pca_data$sdev^2
propVar <- var / sum(var)
cumulative <- cumsum(propVar)
df <- data.frame(comp = seq(1:ncol(cancer %>% select(-id, -diagnosis))), propVar, cumulative)

ggplot(df, aes(x = comp, y = cumulative)) + 
  geom_point() 

### Correlation between pca and original Variables
library(factoextra)
library(corrplot)
all_var <- get_pca_var(pca_data)
corrplot(all_var$contrib, is.corr=FALSE) 

## Contrbutions to PC1 and PC2
library(gridExtra)
p1 <- fviz_contrib(pca_data, choice="var", axes=1, fill="pink", color="grey", top=10)
p2 <- fviz_contrib(pca_data, choice="var", axes=2, fill="skyblue", color="grey", top=10)
grid.arrange(p1,p2,ncol=2)

pca_data$x
### Create data with top 10 PC's
pca_df <- as.data.frame(pca_data$x[,1:10])
############# Create Machine Learning Models ##############
library(caret)
set.seed(1815)
pca <- cbind(diagnosis = cancer$diagnosis, pca_df)
index <- createDataPartition(pca$diagnosis, times = 1, p = 0.8, list = FALSE)
training <- pca[index, ]
testing <-  pca[-index, ]
df_control <- trainControl(method="cv",
                           number = 10,
                           classProbs = TRUE)


############ Apply Machine Learning Models
###1. Logistic Regression Model 
logit <- train(diagnosis ~., data = training, method = "glm", 
               metric = "ROC", preProcess = c("scale", "center"), 
               trControl = df_control)

logit_pred <- predict(logit, testing)
cm_logit <- confusionMatrix(logit_pred, testing$diagnosis, positive = "M")
cm_logit

###2. Random Forest
tuneGrid <- expand.grid(.mtry = c(1: 10))
rf <- train(diagnosis ~., data = training,
            method = "rf", 
            metric = 'Accuracy',
            tuneGrid = tuneGrid,
            trControl = df_control)

rf_pred <- predict(rf, testing)
cm_rf <- confusionMatrix(rf_pred, testing$diagnosis, positive = "M")
cm_rf

# estimate variable importance
importance <- varImp(rf, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


#################### 3. XGBoost ########################################## 
#preparing matrix 
library(xgboost)
dtrain <- xgb.DMatrix(as.matrix(training[,-1]), label = as.numeric(training$diagnosis)-1) 
dtest <- xgb.DMatrix(as.matrix(testing[,-1]), label=as.numeric(testing$diagnosis)-1)

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
               subsample=1, colsample_bytree=1, metric = "auc")
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 30, 
                 nfold = 5, showsd = T, stratified = T, maximize = F)

bestIter = which.min(xgbcv$evaluation_log$test_error_mean)

xgb1 <- xgb.train (params = params, data = dtrain, nrounds = bestIter, 
                   watchlist = list(val=dtest,train=dtrain),eval_metric = "auc")
#model prediction
xgbpred <- predict(xgb1,dtest)
xgb_pred <- as.numeric(xgbpred > 0.5)
cm_xgboost <- confusionMatrix(xgb_pred, as.numeric(testing$diagnosis)-1)
cm_xgboost

# Variable Importance
mat <- xgb.importance(feature_names = colnames(dtrain),model = xgb1)
xgb.plot.importance (importance_matrix = mat, top_n = 5) 

 

