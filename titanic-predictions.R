# Load MASS library that contains QDA
require(MASS)
require(corrplot)
require(ROCR)
require(e1071)

# Load the training and testing data
setwd("D:\\code\\R\\Titanic-Machine-Learning")
train.csv=read.csv("data\\train_new.csv", header=T)
test.csv=read.csv("data\\test_new.csv", header=T)

# Convert the string variable to numeric
train.csv["Title"] = sapply(train.csv["Title"], as.numeric)
train.csv["Sex"] = sapply(train.csv["Sex"], as.numeric)
train.csv["Embarked"] = sapply(train.csv["Embarked"], as.numeric)

test.csv["Title"] = sapply(test.csv["Title"], as.numeric)
test.csv["Sex"] = sapply(test.csv["Sex"], as.numeric)
test.csv["Embarked"] = sapply(test.csv["Embarked"], as.numeric)


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

Log_Fare = log(test.csv[,"Fare"]) 
Log_Fare[is.infinite(Log_Fare)] = 0
hist(Log_Fare)
test.csv = cbind(test.csv, data.frame(Log_Fare))


## Data Visualization
# PCA
num_test_rows = nrow(test.csv[,!names(test.csv) %in% omitted_col])
num_train_rows = nrow(train.csv[,!names(train.csv) %in% omitted_col])

all.data = rbind(train.csv[,!names(train.csv) %in% omitted_col], test.csv[,!names(test.csv) %in% omitted_col]) 
pca = prcomp(all.data[,!names(all.data) %in% omitted_col], center=T, scale=T)
colors =c("green", "red")
par(mfrow=c(1,1))

plot(pca$x[,1]^2, pca$x[,1],col=colors[train.csv$Survived+1])
plot(pca$x[num_train_rows,1]^0.5, pca$x[,2],col=colors[train.csv$Survived+1])
train.csv.pca1 = data.frame(x=cbind(pca$x[1:num_train_rows,1]^2, pca$x[1:num_train_rows,1]), Survived=train.csv$Survived)
train.csv.pca2 = data.frame(x=cbind(pca$x[1:num_train_rows,1]^0.5, pca$x[1:num_train_rows,2]), Survived=train.csv$Survived)
test.csv.pca2 = data.frame(x=cbind(pca$x[num_train_rows+1:num_test_rows,1]^0.5, pca$x[num_train_rows+1:num_test_rows,2]))
plot(train.csv[,predictors], col=colors[train.csv$Survived+1])

# Perform train & test split
test_rows = sample(nrow(train.csv), 100)
test.csv = train.csv[test_rows, ]
train.csv = train.csv[-test_rows, ]

test_rows_pca = sample(nrow(train.csv.pca1), 100)
test.csv.pca1 = train.csv.pca1[test_rows_pca, ]
train.csv.pca1 = train.csv.pca1[-test_rows_pca, ]

test_rows_pca = sample(nrow(train.csv.pca2), 100)
test.csv.pca2 = train.csv.pca2[test_rows_pca, ]
train.csv.pca2 = train.csv.pca2[-test_rows_pca, ]

# Fit Logistic Regression
logit.fit = glm(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked, data=train.csv, family=binomial)
logit.fit.all = glm(Survived ~ .- PassengerId - Survived - Ticket - Name - Title - FamSize, data=train.csv, family=binomial)

# Fit Quadratic Discriminant Analysis
qda.fit = qda(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked, data=train.csv)
qda2.fit = qda(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked + Log_Fare, data=train.csv)

# Fit Support Vector Machine (SVM)
svmfit = svm(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked + Log_Fare, data=train.csv, kernel ="linear", cost=10, scale=FALSE)
svmfit_pca1 = svm(Survived ~ ., data=train.csv.pca1, kernel ="linear", cost=10, scale=FALSE)

# Find best cost value 
tune.out = tune(svm, Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked + Log_Fare,
                data=train.csv,
                kernel ="linear",
                ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))

summary(tune.out)
bestmod = tune.out$best.model
summary(bestmod)

tune.out.pca1 = tune(svm, Survived ~ ., data=train.csv.pca1, kernel ="linear", ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
summary(tune.out.pca1)
bestmod.pca1 = tune.out.pca1$best.model
summary(bestmod.pca1)


tune.out.pca2 = tune(svm, Survived ~ .,
                     data=train.csv.pca2,
                     kernel ="radial",
                     ranges=list(cost=c(0.00001, 0.0001, 0.001, 0.01, 0.1, 1,5,10,100)),
                     gamma=c(0.5,1,2,3,4))
summary(tune.out.pca2)
bestmod.pca2 = tune.out.pca2$best.model
summary(bestmod.pca2)


## Cross Evaluation
# Logistic Regression
# Cross validation
logit.pred = predict(logit.fit, test.csv)
logit.class = ifelse(logit.pred>0.5, 1, 0)
table(logit.class, test.csv$Survived)

# Using all predictors 
logit.pred.all = predict(logit.fit.all, test.csv)
logit.class.all = ifelse(logit.pred.all>0.5, 1, 0)
table(logit.class.all, test.csv$Survived)

# QDA
qda.pred = predict(qda.fit, test.csv)
table(qda.pred$class, test.csv$Survived)
# Adding the log only increases accuracy by a little bit so not worth adding it
qda2.pred = predict(qda2.fit, test.csv)
table(qda2.pred$class, test.csv$Survived)

## SVM Cross Evaluation
svm.pred = predict(svmfit, test.csv, values=T)
svm.pred.rounded = round(svm.pred)
svm.pred.rounded[svm.pred.rounded<0] = 0
table(svm.pred.rounded, test.csv$Survived)

svm_pca.pred = predict(svmfit_pca1, test.csv.pca1, values=T)
svm_pca.pred.rounded = round(svm_pca.pred)
svm_pca.pred.rounded[svm_pca.pred.rounded<0] = 0
table(svm_pca.pred.rounded, test.csv$Survived)

svm.best.pred = predict(bestmod, test.csv)
svm.best.pred.rounded = round(svm.best.pred)
svm.best.pred.rounded[svm.best.pred.rounded<0] = 0
table(predict=svm.best.pred.rounded, truth=test.csv$Survived)

svm.best.pred.pca1 = predict(bestmod.pca1, test.csv.pca1)
svm.best.pred.rounded = round(svm.best.pred.pca1)
svm.best.pred.rounded[svm.best.pred.rounded<0] = 0
table(predict=svm.best.pred.rounded, truth=test.csv.pca1$Survived)

svm.best.pred.pca2 = predict(bestmod.pca2, test.csv.pca2)
svm.best.pred.rounded = round(svm.best.pred.pca2)
svm.best.pred.rounded[svm.best.pred.rounded<0] = 0
table(predict=svm.best.pred.rounded, truth=test.csv.pca2$Survived)


# Function to plot the ROC and compute the AUC
plotROC = function(grouped.pred, test.data) {
  par(mfrow=c(1,1))
  roc_pred = prediction(grouped.pred, test.data$Survived)
  roc_perf = performance(roc_pred, measure = "tpr", x.measure = "fpr")
  plot(roc_perf, main = "ROC", colorize = T)
  abline(a = 0, b = 1)
  auc_perf = performance(roc_pred, measure = "auc")
  auc = auc_perf@y.values[[1]]
  cat("AUC: ")
  cat(auc)
}

plotROC(logit.pred, test.csv)
plotROC(logit.pred.all, test.csv)
plotROC(qda.pred$posterior[,2], test.csv)
plotROC(qda2.pred$posterior[,2], test.csv)
plotROC(svm.pred, test.csv)
plotROC(svm.best.pred, test.csv)
plotROC(svm.best.pred.pca1, test.csv.pca1)
plotROC(svm.best.pred.pca2, test.csv.pca2)

# Kaggle 
svm.pred = predict(bestmod.pca2, test.csv.pca2)
svm.pred.rounded = round(svm.pred)
svm.pred.rounded[svm.pred.rounded<0] = 0
length(svm.pred.rounded)
length(test.csv$Survived)
