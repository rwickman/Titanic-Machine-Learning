# Load MASS library that contains QDA
require(MASS)
require(corrplot)
require(ROCR)
require(e1071)

# Load the training and testing data 
setwd("D:\\_Code\\R\\Titanic-Machine-Learning")
train.csv=read.csv("data\\train_new.csv", header=T)
#test.csv=read.csv("data\\test_new.csv", header=T)

# Convert the string variable to numeric
train.csv["Title"] = sapply(train.csv["Title"], as.numeric)
train.csv["Sex"] = sapply(train.csv["Sex"], as.numeric)
train.csv["Embarked"] = sapply(train.csv["Embarked"], as.numeric)


# Find collinearity
omitted_col = c("PassengerId", "Survived", "Ticket", "Name")
corrplot(cor(train.csv[,!names(train.csv) %in% omitted_col]), method="circle")
omitted_col = append(omitted_col, c("Pclass", "FamSize", "Title"))
corrplot(cor(train.csv[,!names(train.csv) %in% omitted_col]), method="circle")

predictors = names(train.csv)[!names(train.csv) %in% omitted_col]
# Histogram
par(mfrow=c(3,2))
for (i in 1:length(predictors)){
  hist(train.csv[,predictors[i]], 
       main = predictors[i],
       xlab="")
}

# Log transformation for Fare
Log_Fare = log(train.csv[,"Fare"]) 
Log_Fare[is.infinite(Log_Fare)] = 0
hist(Log_Fare)
train.csv = cbind(train.csv, data.frame(Log_Fare))

# Perform train & test split
test_rows = sample(nrow(train.csv), 100)
test.csv = train.csv[test_rows, ]
train.csv = train.csv[-test_rows, ]

# Fit Logistic Regression
logit.fit = glm(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked, data=train.csv, family=binomial)
logit.fit.all = glm(Survived ~ .- PassengerId - Survived - Ticket - Name, data=train.csv, family=binomial)

# Fit Quadratic Discriminant Analysis
qda.fit = qda(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked, data=train.csv)
qda2.fit = qda(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked + Log_Fare, data=train.csv)

# Fit Support Vector Machine (SVM)
svmfit = svm(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked + Log_Fare, data=train.csv, kernel ="linear", cost=10, scale=FALSE)

# Find best cost value 
tune.out = tune(svm, Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked + Log_Fare,
                data=train.csv,
                kernel ="linear",
                ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))

summary(tune.out)
bestmod = tune.out$best.model
summary(bestmod)

## Logistic Regression
# Cross validation
logit.pred = predict(logit.fit, test.csv)
logit.class = ifelse(logit.pred>0.5, 1, 0)
table(logit.class, test.csv$Survived)
# Using all predictors 
logit.pred.all = predict(logit.fit.all, test.csv)
logit.class.all = ifelse(logit.pred.all>0.5, 1, 0)
table(logit.class.all, test.csv$Survived)


## QDA
qda.pred = predict(qda.fit, test.csv)
table(qda.pred$class, test.csv$Survived)
# Adding the log only increases accuracy by a little bit so not worth adding it
qda2.pred = predict(qda2.fit, test.csv)
table(qda2.pred$class, test.csv$Survived)

# SVM Cross Evaluation
svm.pred = predict(svmfit, test.csv, values=T)
svm.pred.rounded = round(svm.pred)
svm.pred.rounded[svm.pred.rounded<0] = 0
table(svm.pred.rounded, test.csv$Survived)

svm.best.pred = predict(bestmod, test.csv)
svm.best.pred.rounded = round(svm.best.pred)
svm.best.pred.rounded[svm.best.pred.rounded<0] = 0
table(predict=svm.best.pred.rounded, truth=test.csv$Survived)

# Function to plot the ROC and compute the AUC
plotROC = function(grouped.pred) {
  par(mfrow=c(1,1))
  roc_pred = prediction(grouped.pred, test.csv$Survived)
  roc_perf = performance(roc_pred, measure = "tpr", x.measure = "fpr")
  plot(roc_perf, main = "ROC", colorize = T)
  abline(a = 0, b = 1)
  auc_perf = performance(roc_pred, measure = "auc")
  auc = auc_perf@y.values[[1]]
  cat("AUC: ")
  cat(auc)
}

plotROC(logit.pred)
plotROC(logit.pred.all)
plotROC(qda.pred$posterior[,2])
plotROC(qda2.pred$posterior[,2])
plotROC(svm.pred)
plotROC(svm.best.pred)
