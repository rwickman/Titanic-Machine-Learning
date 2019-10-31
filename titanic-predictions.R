# Load the training and testing data 
setwd("D:\\_Code\\R\\Titanic-Machine-Learning")
train.csv=read.csv("data\\train_new.csv", header=T)
#test.csv=read.csv("data\\test_new.csv", header=T)

# Convert the string variable to numeric
train.csv["Title"] = sapply(train.csv["Title"], as.numeric)
train.csv["Sex"] = sapply(train.csv["Sex"], as.numeric)
train.csv["Embarked"] = sapply(train.csv["Embarked"], as.numeric)


# Perform train & test split
test_rows = sample(nrow(train.csv), 100)
test.csv = train.csv[test_rows, ]
train.csv = train.csv[-test_rows, ]


# Load MASS library that contains QDA
require(MASS)
require("corrplot")


# Find collinearity
omitted_col = c("PassengerId", "Survived", "Ticket", "Name")
corrplot(cor(train.csv[,!names(train.csv) %in% omitted_col]), method="circle")
omitted_col = append(omitted_col, c("Pclass", "FamSize", "Title"))
corrplot(cor(train.csv[,!names(train.csv) %in% omitted_col]), method="circle")

# Fit Quadratic Discriminant Analysis
predictors = names(train.csv)[!names(train.csv) %in% omitted_col]
predictors
#qda.fit = qda(formula(paste("Survived ~ ", paste(predictors, collapse=" + "))), data=train.csv)
qda.fit = qda(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked, data=train.csv)

# Cross validation
qda.class = predict(qda.fit, test.csv)$class
#which( == data.frame(qda.pred))
table(qda.pred, test.csv$Survived)