library(openxlsx)
dat<-openxlsx::read.xlsx("D:/Education/Competition/AV/NMIMS Hyderabad/Round 3/Data Hack Round 3 Case Study/Copy of Housing_Resale_Data (1).xlsx", sheet= 1)
#dat<-backup

# Check for duplicated rows
dat<-dat[!duplicated(dat),] ### 803 duplicate values

########################### Exploratory Data Analysis ###################################
# unique values of all column 
sapply(dat[,-10],function(x) length(unique(x)))
# month                town           flat_type               block 
# 215                  26                   7                2190 
# street_name        storey_range      floor_area_sqm          flat_model 
# 534                  25                 191                  20 
# lease_commence_date 
# 50 

# Resale Price 
#ggplot(dat,aes(x=1,y=resale_price))+geom_boxplot(fill = "steelblue")

### Outliers in resale_price - have to use median

# Month Column
median<-as.data.frame(tapply(dat$resale_price,list(month=dat$month),median))
colnames(median)<-c("Price_Median")
median$Month<-rownames(median)
barplot(median$Price_Median,col = "steelblue", xlab = "Month")

# Month Column
length(unique(dat$month)) # 215 --- 2000-01 to 2017-11
dat$year<-sapply(dat$month, function(x) unlist(strsplit(x,"-"))[1])
dat$Month<-sapply(dat$month, function(x) unlist(strsplit(x,"-"))[2])

## Create difference of present ot lease date
dat$yearFromLease<-as.integer(dat$year) - as.integer(dat$lease_commence_date)
dat<-dat[,!names(dat) %in% c("lease_commence_date")]
dat$floor_area_sqm[dat$floor_area_sqm>quantile(dat$floor_area_sqm,0.99)]<- quantile(dat$floor_area_sqm,0.99)
dat$yearFromLease[dat$yearFromLease<4]<-3
dat$yearFromLease[dat$yearFromLease>42]<-42

# Univariate Analysis
### FLoor Area Plot 
ggplot(dat,aes(x=log(floor_area_sqm),y=log(resale_price)))+geom_point(fill = "steelblue")
summary(lm(log(resale_price)~log(floor_area_sqm),dat))

## resale price per square feet
dat$pricePerMeter<-dat$resale_price/dat$floor_area_sqm
summary(lm(log(resale_price)~log(pricePerMeter),dat))
dat$log_pricePerMeter<-log(dat$pricePerMeter)

### year from lease
ggplot(dat,aes(x=yearFromLease,y=log(resale_price)))+geom_point(fill = "steelblue")
summary(lm(resale_price~yearFromLease,dat))
# Create category
library(rpart)
library(rpart.plot)
d<-rpart(resale_price~yearFromLease,data=dat)
rpart.plot(d)

dat$yearFromLease<-ifelse(dat$yearFromLease<14,"New","old")
# 5% of the variabilty explained by year from lease

# Month
dat$year<-as.numeric(dat$year)
dat$Month<-as.numeric(dat$Month)
summary(lm(log(resale_price)~log(Month),dat)) # only 0.02% 
summary(lm(log(resale_price)~year,dat)) # 36% 

##### Use all numeric function to predict resale Price
dat$log_resale_price<-log(dat[,"resale_price"])
dat$log_floorArea<-log(dat[,"floor_area_sqm"])

library(ggplot2)
require(GGally)
plot1<-ggpairs(data=dat, columns=c("yearFromLease","resale_price","floor_area_sqm","pricePerMeter"),
               mapping = aes(color = "dark green"),
               axisLabels="show")
plot1
summary(lm(log(resale_price)~year+yearFromLease+log(floor_area_sqm),data=dat))

############################### Categorical Variable #########################
# Street Name # top 10 expensive street
str<-as.data.frame(tapply(dat$resale_price,list(streetname=dat$street_name),median))
str$streetName<-rownames(str)
colnames(str)<-c("Resale_Price_Median","streetName")
str<-str[order(-str$Resale_Price_Median),]
top_10_streets<-str[1:20,]

# Town # top 10 expensive town
str<-as.data.frame(tapply(dat$resale_price,list(town=dat$town),median))
str$town<-rownames(str)
colnames(str)<-c("Resale_Price_Median","Town")
str<-str[order(-str$Resale_Price_Median),]
top_10_town<-str[1:10,]
# Price distribution across town
str<-as.data.frame(table(dat$town))
str<-str[order(-str$Freq),]
to<-str[(1:10),]
sub<-dat[dat$town %in% to$Var1,]
summary(lm(log(resale_price)~town,data= dat))
ggplot(dat[dat$town %in% top_10_town$Town,],aes(x=town,y=resale_price))+geom_boxplot(fill = "steelblue")


# block # top 10 expensive block
str<-as.data.frame(tapply(dat$resale_price,list(block=dat$block),median))
str$block<-rownames(str)
colnames(str)<-c("Resale_Price_Median","block")
str<-str[order(-str$Resale_Price_Median),]
top_10_block<-str[1:10,]

## Flat Type 
g1<-ggplot(dat,aes(x=flat_type,y=dat$resale_price))+geom_boxplot(fill = "steelblue")+ylab("Resale Price")
g2<-ggplot(dat,aes(x=flat_type,y=dat$pricePerMeter))+geom_boxplot(fill = "pink")+ylab("Price Per Meter")
#install.packages("gridExtra")
library(gridExtra)
grid.arrange(g1, g2, ncol=2)
## Flat Model
str<-as.data.frame(tapply(dat$resale_price,list(town=dat$flat_model),median))
str$flat_model<-rownames(str)
colnames(str)<-c("Resale_Price_Median","FlatModel")
str<-str[order(-str$Resale_Price_Median),]
top_10_Model<-str[1:10,]
sub<-dat[dat$flat_model %in% top_10_Model$FlatModel,]
g1<-ggplot(dat,aes(x=flat_model,y=resale_price))+geom_boxplot(fill = "steelblue")+ylab("Resale Price")
g1
table(dat$flat_model)
dat$flat_model[dat$flat_model %in% c("DBSS","Adjoined flat",
                                             "Model A-Maisonette",
                                             "Multi Generation",
                                             "Premium Apartment Loft",
                                             "Premium Maisonette",
                                             "Type S1","Type S2")]<-"ExpensiveModel"
dat$flat_model[dat$flat_model %in% c("ExpensiveModel","Terrace")]<-"ExpensiveModel"
dat$flat_model[dat$flat_model %in% c("Maisonette","Improved-Maisonette")]<-"Maisonette"
dat$flat_model[dat$flat_model %in% c("2-room","Simplified")]<-"Simplified"

                                     
### Tackle with Town, Street and Block (by finding the region to whom they belong)
################################################# Apply model only on numeric variable

#SEMBAWANG - > northernmost tip of the North Region of Singapore. -> North
#KALLANG/WHAMPOA -> the Central Region of Singapore -> Centeral
#GEYLANG -> Central Region of Singapore. -> Central
#SENGKANG -> North-East Region of Singapore -> Northe-East
# ANG MO KIO -> North-East Region of Singapore -> North-East
# CENTRAL AREA -> CENTRAL AREA -> Central 
# BUKIT PANJANG -> West Region of Singapore -> West 
# BEDOK -> East Region of Singapore -> East
# BUKIT MERAH -> southernmost part of the Central Region of Singapore. -> Central
# BISHAN -> northernmost portion of the Central Region of Singapore -> Central
# JURONG WEST -> West Region of Singapore -> West
# BUKIT BATOK -> eastern boundary of the West Region of Singapore -> West
# Hougang -> North-East Region of Singapore -> North East
# QUEENSTOWN -> south-westernmost fringe of the Central Region of Singapore -> Cental
# CLEMENTI -> West Region of Singapore -> West
# PASIR RIS -> East Region of Singapore -> East
# CHOA CHU KANG -> north-westernmost point of the West Region of Singapore -> West
# PUNGGOL ->  North-East Region of Singapore -> North-east
# BUKIT TIMAH -> westernmost part of the Central Region of Singapore -> Central
# TOA PAYOH -> northern part of the Central Region of Singapore -> Central 
# JURONG EAST -> West Region of Singapore -> West
# SERANGOON ->  North-East Region of Singapore -> North East
# MARINE PARADE -> Central Region of Singapore -> Central
# WOODLANDS -> North Region of Singapore -> North
# TAMPINES-> East Region of Singapore -> East
# YISHUN -> northeastern corner of the North Region of Singapore, -> North

#North - 3 , West - 6, East - 3 , Central - 9 ,Nort-East - 5

central <- c("CENTRAL AREA","KALLANG/WHAMPOA","GEYLANG","BUKIT MERAH","BISHAN","QUEENSTOWN","BUKIT TIMAH","TOA PAYOH","MARINE PARADE")
north<-c("SEMBAWANG","WOODLANDS","YISHUN")
east<-c("BEDOK","PASIR RIS","TAMPINES")
north_east<-c("ANG MO KIO","SENGKANG","Hougang","PUNGGOL","SERANGOON")
dat$Region<-ifelse(dat$town %in% central,"Central",ifelse(dat$town %in% north,"North",ifelse(dat$town %in% east,"East",
                                                                                             ifelse(dat$town %in% north_east,"North_East","West"))))
t<-as.data.frame(table(dat$Region))
ggplot(t, aes(x = Var1,y = Freq)) + geom_bar(stat="identity",fill="steelblue")+
  geom_text(aes(label=Freq), vjust=1.6, color="red", size=3.5)+
  xlab("Region")

# distribution of sales price across region
ggplot(dat,aes(x=resale_price))+geom_histogram(fill="steelblue",bins = 10)+facet_grid(~Region)+theme_bw()

# Affect of different streetname, block, town and region on resale price
summary(lm(log(resale_price)~street_name, dat)) #R^2 = 28% 
summary(lm(log(resale_price)~town, dat)) # *%
summary(lm(log(resale_price)~block, dat)) # error out of memory
summary(lm(log(resale_price)~Region, dat)) # 1.5%

# Visualize Street Name (extracted lat and long)
geo<-read.csv("D:/Education/geo.csv")
n<-data.frame(resale_price=dat$resale_price,street=dat$street_name)
d<-merge(n,geo,by.x="street",by.y="street_name",all=T)
summary(lm(log(resale_price)~Lat+Long, d)) #R^2 = 28% 
# VIsualization
library(rworldmap)
library(ggmap)
library(ggplot2)
d$color<-ifelse(d$street %in% top_10_streets$streetName, "red","blue")
sing <- get_map(location = "singapore", color = "bw",
                zoom = 11, maptype = "toner", source = "google")
ggmap(sing) +
  geom_point(data = d, aes(x = Long, y = Lat,col=color))

## combine Street categories based on price per meter 
dat<-within(dat, { street_pricePMMed<- as.numeric(ave(pricePerMeter,street_name, FUN=function(x) median(x)))})
summary(dat$street_pricePMMed)
dat$Street_newPPM<-ifelse(dat$street_pricePMMed<2562,"Low_Price",
                          ifelse(dat$street_pricePMMed>=2562 & dat$street_pricePMMed<2858,"Moderate_Price",
                          ifelse(dat$street_pricePMMed>3243,"Most_Expensive","Expensive")))
summary(lm(log(resale_price)~Street_newPPM,data=dat))

## combine town categories based on price per meter 
dat<-within(dat, { town_pricePMMed<- as.numeric(ave(pricePerMeter,town, FUN=function(x) median(x)))})
summary(dat$town_pricePMMed)
dat$town_newPPM<-ifelse(dat$town_pricePMMed<2718,"Low_Price",
                          ifelse(dat$town_pricePMMed>=2921 & dat$street_pricePMMed<2858,"Moderate_Price",
                                 ifelse(dat$town_pricePMMed>3093,"Most_Expensive","Expensive")))
summary(lm(log(resale_price)~town_newPPM,data=dat))

## combine block categories based on price per meter 
dat<-within(dat, { block_pricePMMed<- as.numeric(ave(pricePerMeter,block, FUN=function(x) median(x)))})
summary(dat$block_pricePMMed)
dat$block_newPPM<-ifelse(dat$block_pricePMMed<2734,"Low_Price",
                        ifelse(dat$block_pricePMMed>=2899 & dat$street_pricePMMed<2858,"Moderate_Price",
                               ifelse(dat$block_pricePMMed>3147,"Most_Expensive","Expensive")))
summary(lm(log(resale_price)~block_newPPM,data=dat))

## Create lm model
sigVars<-c("flat_type","storey_range","flat_model","year","Month",
           "yearFromLease","Region","log_floorArea","Street_newPPM",
           "town_newPPM" ,"block_newPPM","log_resale_price")
df<-dat[,names(dat) %in% sigVars]
df <- as.data.frame(model.matrix( ~ .-1, df))
# Split data into train and test
smp_size<-floor(0.8*nrow(df))
set.seed(123)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
train <- df[train_ind, ]
test <- df[-train_ind, ]
# Correlated Variable (iteratively removed)
lmodel<-lm(log_resale_price~.,data=train)
library(car)
vif(lmodel)
summary(lmodel)

corvar<-c("flat_typeMULTI-GENERATION","flat_type4 ROOM","flat_modelModel A")
train<-train[,!names(train) %in% corvar]
test<-test[,!names(test) %in% corvar]

################################################## Build linear reg
lmodel<-lm(log_resale_price~.,data=train)
summary(lmodel)
pred<-predict(lmodel,newdata=test[,!names(test) %in% c("log_resale_price")])

# Evaluate
library(Metrics)
rmse(exp(test$log_resale_price),exp(pred))
# 58794.68
################################################## Lasso Regression
x_train<-train[,!names(train) %in% c("log_resale_price")]
y<-train$log_resale_price
library(glmnet)
crossval <-  cv.glmnet(x = as.matrix(x_train), y = y)
plot(crossval)
penalty <- crossval$lambda.min #optimal lambda
penalty #minimal shrinkage
fit1 <-glmnet(x = as.matrix(x_train), y = y, alpha = 1, lambda = penalty ) #estimate the model with that
coef(fit1)

################################################################################# XGboost Model
library(xgboost)
train$floor_area_sqm<-exp(train$log_floorArea)
test$floor_area_sqm<-exp(test$log_floorArea)

train$resale_price<-exp(train$log_resale_price)
test$resale_price<-exp(test$log_resale_price)

x_train<-train[,!names(train) %in% c("log_resale_price","resale_price","log_floorArea")]
y<-train$resale_price
## Tuned using cross validation(cv.xgb())
xgbFit_complete = xgboost(data = as.matrix(x_train), nfold = 4, label = y, 
                          nrounds = 2000, verbose = T, objective = "reg:linear", eval_metric = "rmse", 
                          nthread = 8, eta = 0.1, gamma = 0.04, max_depth = 6, min_child_weight = 1.78, 
                          subsample = 0.5, colsample_bytree = 0.8)

## Predictions
x_test<-test[,!names(test) %in% c("log_resale_price","resale_price","log_floorArea")]
preds2 <- predict(xgbFit_complete, newdata = as.matrix(x_test))
rmse(test$resale_price, preds2)
#[1] 26385.99

## Variable Importance
model <- xgb.dump(xgbFit_complete, with_stats = T)
model[1:20] #This statement prints top 10 nodes of the model

# Get the feature real names
names <- dimnames(train[,!names(train) %in% c("resale_price")])[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgbFit_complete)
# Nice graph
xgb.plot.importance(importance_matrix[importance_matrix$Importance>0.001,])

backup<-dat
# Feature Engineering
## town wise
dat<-within(dat, { flat_type_townwise <- as.numeric(ave(flat_type, town, FUN=function(x) length(unique(x))))})
dat<-within(dat, { block_townwise <- as.numeric(ave(block, town, FUN=function(x) length(unique(x))))})
dat<-within(dat, { street_townwise <- as.numeric(ave(street_name, town, FUN=function(x) length(unique(x))))})
dat<-within(dat, { storeyRange_townwise <- as.numeric(ave(storey_range, town, FUN=function(x) length(unique(x))))})
dat<-within(dat, { flat_model_townwise <- as.numeric(ave(flat_model, town, FUN=function(x) length(unique(x))))})

## Block wise
dat<-within(dat, { flat_type_blockwise <- as.numeric(ave(flat_type, block, FUN=function(x) length(unique(x))))})
dat<-within(dat, { street_name_blockwise <- as.numeric(ave(street_name, block, FUN=function(x) length(unique(x))))})
dat<-within(dat, { storey_range_blockwise <- as.numeric(ave(storey_range, block, FUN=function(x) length(unique(x))))})
dat<-within(dat, { flat_model_blockwise <- as.numeric(ave(flat_model, block, FUN=function(x) length(unique(x))))})

## Streetwise
dat<-within(dat, { flat_type_streetwise <- as.numeric(ave(flat_type, street_name, FUN=function(x) length(unique(x))))})
dat<-within(dat, { block_streetwise <- as.numeric(ave(block, street_name, FUN=function(x) length(unique(x))))})
dat<-within(dat, { storey_range_streetwise <- as.numeric(ave(storey_range, street_name, FUN=function(x) length(unique(x))))})
dat<-within(dat, { flat_model_streetwise <- as.numeric(ave(flat_model, street_name, FUN=function(x) length(unique(x))))})

## flat_type
dat<-within(dat, { flat_model_flatTypewise <- as.numeric(ave(flat_model, flat_type, FUN=function(x) length(unique(x))))})
dat<-within(dat, { town_flatTypewise <- as.numeric(ave(town, flat_type, FUN=function(x) length(unique(x))))})
dat<-within(dat, { block_flatTypewise <- as.numeric(ave(block, flat_type, FUN=function(x) length(unique(x))))})
dat<-within(dat, { street_flatTypewise <- as.numeric(ave(street_name, flat_type, FUN=function(x) length(unique(x))))})
dat<-within(dat, { storey_range_flatTypewise <- as.numeric(ave(storey_range, flat_type, FUN=function(x) length(unique(x))))})

## Storey_Range
dat<-within(dat, { town_storeywise <- as.numeric(ave(town,storey_range, FUN=function(x) length(unique(x))))})
dat<-within(dat, { block_storeywise <- as.numeric(ave(block,storey_range, FUN=function(x) length(unique(x))))})
dat<-within(dat, { flat_type_storeywise <- as.numeric(ave(flat_type,storey_range, FUN=function(x) length(unique(x))))})
dat<-within(dat, { flat_model_storeywise <- as.numeric(ave(flat_model,storey_range, FUN=function(x) length(unique(x))))})
dat<-within(dat, { street_storeywise <- as.numeric(ave(street_name,storey_range, FUN=function(x) length(unique(x))))})

## Flat Model
dat<-within(dat, { town_flatModelwise <- as.numeric(ave(town,flat_model, FUN=function(x) length(unique(x))))})
dat<-within(dat, { street_name_flatModelwise <- as.numeric(ave(street_name,flat_model, FUN=function(x) length(unique(x))))})
dat<-within(dat, { block_flatModelwise <- as.numeric(ave(block,flat_model, FUN=function(x) length(unique(x))))})
dat<-within(dat, { flat_type_flatModelwise <- as.numeric(ave(flat_type,flat_model, FUN=function(x) length(unique(x))))})
dat<-within(dat, { storey_flatModelwise <- as.numeric(ave(storey_range,flat_model, FUN=function(x) length(unique(x))))})

# Min, Max, median of area
dat<-within(dat, { storey_AreaMed<- as.numeric(ave(floor_area_sqm,storey_range, FUN=function(x) median(x)))})
dat<-within(dat, { town_AreaMed<- as.numeric(ave(floor_area_sqm,town, FUN=function(x) median(x)))})
dat<-within(dat, { flat_type_AreaMed<- as.numeric(ave(floor_area_sqm,flat_type, FUN=function(x) median(x)))})
dat<-within(dat, { flat_model_AreaMed<- as.numeric(ave(floor_area_sqm,flat_model, FUN=function(x) median(x)))})
dat<-within(dat, { block_AreaMed<- as.numeric(ave(floor_area_sqm,block, FUN=function(x) median(x)))})
dat<-within(dat, { street_name_AreaMed<- as.numeric(ave(floor_area_sqm,street_name, FUN=function(x) median(x)))})
dat<-within(dat, { storey_AreaMin<- as.numeric(ave(floor_area_sqm,storey_range, FUN=function(x) min(x)))})
dat<-within(dat, { town_AreaMin<- as.numeric(ave(floor_area_sqm,town, FUN=function(x) min(x)))})
dat<-within(dat, { flat_type_AreaMin<- as.numeric(ave(floor_area_sqm,flat_type, FUN=function(x) min(x)))})
dat<-within(dat, { flat_model_AreaMin<- as.numeric(ave(floor_area_sqm,flat_model, FUN=function(x) min(x)))})
dat<-within(dat, { block_AreaMin<- as.numeric(ave(floor_area_sqm,block, FUN=function(x) min(x)))})
dat<-within(dat, { street_name_AreaMin<- as.numeric(ave(floor_area_sqm,street_name, FUN=function(x) min(x)))})
dat<-within(dat, { storey_AreaMax<- as.numeric(ave(floor_area_sqm,storey_range, FUN=function(x) max(x)))})
dat<-within(dat, { town_AreaMax<- as.numeric(ave(floor_area_sqm,town, FUN=function(x) max(x)))})
dat<-within(dat, { flat_type_AreaMax<- as.numeric(ave(floor_area_sqm,flat_type, FUN=function(x) max(x)))})
dat<-within(dat, { flat_model_AreaMax<- as.numeric(ave(floor_area_sqm,flat_model, FUN=function(x) max(x)))})
dat<-within(dat, { block_AreaMax<- as.numeric(ave(floor_area_sqm,block, FUN=function(x) mean(x)))})
dat<-within(dat, { street_name_AreaMax<- as.numeric(ave(floor_area_sqm,street_name, FUN=function(x) max(x)))})

# Frequency of categories (block and Street Name)
# town 
backup <- dat
dat<-backup
## Building Linear Regression
d<-cbind(df,dat[,c(4,5,7,9,24:69)])
##count variable
tab<-as.data.frame(table(dat$block),stringsAsFactors = F)
colnames(d)<-c("Var1","BlockCount")
demo<-merge(d,tab,by.x = "block" ,by.y = "Var1",all=T)

## Street Name
tab<-as.data.frame(table(dat$street_name),stringsAsFactors = F)
colnames(d)<-c("Var1","StreetCount")
dat<-merge(demo,tab,by.x = "street_name" ,by.y = "Var1",all=T)
##############################################################################
corr<-as.data.frame(cor(dat[,c(9,24:69)]))
write.csv(corr,"D:/Education/corr.csv")

corrVars<-c("Region_AreaMax","street_name_blockwise","block_streetwise",
            "storey_range_flatTypewise","street_flatTypewise",
            "street_storeywise","street_name_flatModelwise",
            "flat_type_AreaMed","block_AreaMax","flat_type_AreaMin",
            "block_storeywise","block_flatModelwise","storey_flatModelwise",
            "Room2_Flat","Region_AreaMed","storey_AreaMin",
            "flat_type_storeywise","flat_model_AreaMin","Region_AreaMin",
            "flat_model_blockwise","flat_model_flatTypewise","flat_model_storeywise",
            "MultiGeneration_Flat","Room3_Flat","Room4_Flat")
d<-dat
d<-d[,!names(d) %in% corrVars]

## XGBoost with all derived vriables
library(xgboost)
train <- d[train_ind, ]
test <- d[-train_ind, ]

x_train<-train[,!names(train) %in% c("block","street_name","log_resale_price","resale_price","log_floorArea")]
y<-train$resale_price
## Tuned using cross validation(cv.xgb())
xgbFit_complete = xgboost(data = as.matrix(x_train), nfold = 4, label = y, 
                          nrounds = 2000, verbose = T, objective = "reg:linear", eval_metric = "rmse", 
                          nthread = 8, eta = 0.1, gamma = 0.04, max_depth = 6, min_child_weight = 1.78, 
                          subsample = 0.5, colsample_bytree = 0.8)

## Predictions
x_test<-test[,!names(test) %in% c("block","street_name","log_resale_price","resale_price","log_floorArea")]
preds2 <- predict(xgbFit_complete, newdata = as.matrix(x_test))
rmse(test$resale_price, preds2)
#[1] 17679.65

## Variable Importance
# Get the feature real names
names <- dimnames(x_train[,!names(x_train) %in% c("resale_price")])[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgbFit_complete)
# Nice graph
xgb.plot.importance(importance_matrix)

## Extract Important Variables
importance_matrix
sigVars<-importance_matrix[importance_matrix$Importance>0.001,"Feature"]

# Run only on significant vars
x_train<-x_train[,names(x_train) %in% sigVars$Feature]
y<-train$resale_price
## Tuned using cross validation(cv.xgb())
xgbFit_complete = xgboost(data = as.matrix(x_train), nfold = 4, label = y, 
                          nrounds = 2000, verbose = T, objective = "reg:linear", eval_metric = "rmse", 
                          nthread = 8, eta = 0.1, gamma = 0.04, max_depth = 6, min_child_weight = 1.78, 
                          subsample = 0.5, colsample_bytree = 0.8)

## Predictions
x_test<-test[,names(test) %in% sigVars$Feature]
preds2 <- predict(xgbFit_complete, newdata = as.matrix(x_test))
rmse(test$resale_price, preds2)
#17625.67

## Variable Importance
# Get the feature real names
names <- dimnames(x_train)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgbFit_complete)
# Nice graph
xgb.plot.importance(importance_matrix)
importance_matrix

## Select further variables
importance_matrix
sigVars2<-importance_matrix[importance_matrix$Importance>0.002,"Feature"]

# Run only on significant vars
x_train<-x_train[,names(x_train) %in% sigVars2$Feature]
y<-train$resale_price
## Tuned using cross validation(cv.xgb())
xgbFit_complete = xgboost(data = as.matrix(x_train), nfold = 4, label = y, 
                          nrounds = 2000, verbose = T, objective = "reg:linear", eval_metric = "rmse", 
                          nthread = 8, eta = 0.1, gamma = 0.04, max_depth = 6, min_child_weight = 1.78, 
                          subsample = 0.5, colsample_bytree = 0.8)

## Predictions
x_test<-x_test[,names(x_test) %in% sigVars2$Feature]
preds2 <- predict(xgbFit_complete, newdata = as.matrix(x_test))
rmse(test$resale_price, preds2)
#17668.82 (by removing 9 variables - RMSE increased from 18027 to 18441)

## Variable Importance
# Get the feature real names
names <- dimnames(x_train)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgbFit_complete)
# Nice graph
xgb.plot.importance(importance_matrix)
importance_matrix

## Select further variables
importance_matrix
sigVars3<-importance_matrix[importance_matrix$Importance>0.005,"Feature"]

# Run only on significant vars
x_train<-x_train[,names(x_train) %in% sigVars3$Feature]
y<-train$resale_price
## Tuned using cross validation(cv.xgb())
xgbFit_complete = xgboost(data = as.matrix(x_train), nfold = 4, label = y, 
                          nrounds = 2000, verbose = T, objective = "reg:linear", eval_metric = "rmse", 
                          nthread = 8, eta = 0.1, gamma = 0.04, max_depth = 6, min_child_weight = 1.78, 
                          subsample = 0.5, colsample_bytree = 0.8)

## Predictions
x_test<-x_test[,names(x_test) %in% sigVars3$Feature]
preds2 <- predict(xgbFit_complete, newdata = as.matrix(x_test))
rmse(test$resale_price, preds2)
#19787.08 

## Variable Importance
# Get the feature real names
names <- dimnames(x_train)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgbFit_complete)
# Nice graph
xgb.plot.importance(importance_matrix)
importance_matrix

### Explore Variables
co<-c("flat_model_AreaMed","resale_price","flat_type_AreaMed","flat_model_flatTypewise")
d<-dat[,names(dat) %in% co]
corr<-as.data.frame(cor(d))
library(ggplot2)
require(GGally)
plot1<-ggpairs(data=d, columns=1:4,
               mapping = aes(color = "dark green"),
               axisLabels="show")
plot1

#################################################################################################
# ## Random Forest
# library(randomForest)
# library(Metrics)
# rf=randomForest(resale_price ~ . , data = train, mtry=5,ntree=10,do.trace = TRUE) 
# preds <- predict(rf, test[,!names(test) %in%  c("resale_price")])
# rmse(test$resale_price, preds)
# 
# # Variable importance by Random Forest
# varImp(rf)
# 
# library(randomForest)
# library(Metrics)
# rf=randomForest(resale_price ~ . , data = train, mtry=5,ntree=50,do.trace = TRUE) 
# preds <- predict(rf, test[,!names(test) %in%  c("resale_price")])
# rmse(test$resale_price, preds)
# 
# # Variable importance by Random Forest
# rf$importance
# importance(rf)
# # ## GBM Model
# # library(caret)
# # set.seed(222)
# ## Set up caret model training parameters
# CARET.TRAIN.CTRL <- trainControl(method = "repeatedcv", number = 5, repeats = 5, 
#                                  verboseIter = FALSE)
# gbmFit <- train(resale_price ~ ., method = "gbm", metric = "RMSE", 
#                 trControl = CARET.TRAIN.CTRL, 
#                 tuneGrid = expand.grid(n.trees = 500, 
#                 interaction.depth = c(5), shrinkage = c(0.05), 
#                 n.minobsinnode = c(10)), 
#                 data = train, verbose = T)
######################################################################################

# ## GBM Model
# x_train<-train[,names(train) %in% sigVars$Feature]
# y<-train$resale_price
# training<-cbind(x_train,y)
# # set caret training parameters
# library(caret)
# control <- trainControl(method = 'repeatedcv',
#                         number = 5,
#                         repeats = 3,
#                         search = 'grid')
# seed <- 7
# set.seed(seed)
# metric <- 'rmse'
# gbm_mod <- train(y~., 
#                  data = training,
#                  method = 'gbm',
#                  metric = metric,
#                  trControl = control)
# 
