# Load the training and testing data 
setwd("D:\\_Code\\R\\Titanic-Machine-Learning")
train.csv=read.csv("data\\train_new.csv", header=T)
test.csv=read.csv("data\\test_new.csv", header=T)


# Load MASS library that contains QDA
require(MASS)
require("corrplot")

# Convert the string variable to numeric
train.csv["Title"] = sapply(train.csv["Title"], as.numeric)
train.csv["Sex"] = sapply(train.csv["Sex"], as.numeric)
train.csv["Embarked"] = sapply(train.csv["Embarked"], as.numeric)

# Find collinearity
omitted_col = c("PassengerId", "Survived", "Ticket", "Name")
corrplot(cor(train.csv[,!names(train.csv) %in% omitted_col]), method="circle")
omitted_col = append(omitted_col, c("Pclass", "FamSize", "Title"))
corrplot(cor(train.csv[,!names(train.csv) %in% omitted_col]), method="circle")

# Quadratic Discriminant Analysis
predictors = names(train.csv)[!names(train.csv) %in% omitted_col]
qda.fit=qda(formula(paste("Survived ~ ", paste(predictors, collapse=" + "))),data=train.csv)
