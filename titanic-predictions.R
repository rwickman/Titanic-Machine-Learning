# Load MASS library that contains QDA
require(MASS)
require("corrplot")
require(ROCR)


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

# Fit Quadratic Discriminant Analysis
qda.fit = qda(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked, data=train.csv)
qda2.fit = qda(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked + Log_Fare, data=train.csv)

# Cross validation
qda.pred = predict(qda.fit, test.csv)
table(qda.pred$class, test.csv$Survived)

# Adding the log only increases accuracy by a little bit so not worth adding it
qda2.pred = predict(qda2.fit, test.csv)
table(qda2.pred$class, test.csv$Survived)


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

plotROC(qda.pred$posterior[,2])
