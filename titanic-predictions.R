rm(list=ls())
# Load libraries
require(caret)
require(keras)
require(neuralnet)
require(MASS)
require(corrplot)
require(ROCR)
require(e1071)


# Make a prediction given a model and test data
# It will also plot the confusion matrix
make_prediction = function(pred.mod, test_data)
{
  pred = predict(pred.mod, test_data)
  pred.rounded = round(pred)
  pred.rounded[pred.rounded<0] = 0
  print(table(predict=pred.rounded, truth=test_data$Survived))
  pred
}

# Function to plot the ROC and compute the AUC
plotROC = function(grouped.pred, test_data) {
  par(mfrow=c(1,1))
  roc_pred = ROCR::prediction(grouped.pred, test_data$Survived)
  roc_perf = performance(roc_pred, measure = "tpr", x.measure = "fpr")
  plot(roc_perf, main = "ROC", colorize = T)
  abline(a = 0, b = 1)
  auc_perf = performance(roc_pred, measure = "auc")
  auc = auc_perf@y.values[[1]]
  cat("AUC: ")
  cat(auc)
  auc
}

# Returns the best svm model
find_best_svm = function(train_data, kernel_type="radial")
{
  if (kernel_type == "radial")
  {
    tune.out = tune(svm, Survived ~ .,
                        data=train_data,
                        kernel=kernel_type,
                        ranges=list(cost=c(0.00001, 0.001, 0.01, 0.1, 1,5,10,20,100)),
                        gamma=c(0.1, 0.5,1,2,3,4, 10,100))  
  }
  else if (kernel_type == "polynomial")
  {
    tune.out = tune(svm, Survived ~ .,
                    data=train_data,
                    kernel=kernel_type,
                    ranges=list(cost=c(0.00001, 0.001, 0.01, 0.1, 1,5,10,20,100)),
                    gamma=1,
                    d=c(0.5,1,2)) 
  }
  else if (kernel_type == "linear")
  {
    tune.out = tune(svm, Survived ~ .,
                         data=train_data,
                         kernel=kernel_type,
                         ranges=list(cost=c(0.001, 0.01, 0.1, 1, 10)))
  }

  summary(tune.out)
  bestmod = tune.out$best.model
  summary(bestmod)
  bestmod
}


# Find the best number of components to use and the corresponding model
find_best_num_components = function(train_data, test_data, kernel_type="radial")
{
  max_components = ncol(train_data) - 1 
  max_auc = 0
  bestmod_overall = NULL
  best_num_components = 0
  aucs = c(1:max_components)
  for (i in 1:max_components)
  {
    print("")
    print(paste("Testing components", i))
    bestmod = find_best_svm(train_data[,c(1:i, ncol(train_data))])
    bestmod_pred = make_prediction(bestmod, test_data)
    auc = plotROC(bestmod_pred, test_data)
    aucs[i] = auc
    if (auc > max_auc)
    {
      max_auc = auc
      bestmod_overall = bestmod
      best_num_components = i
    }
  }
  plot(x=c(1:max_components), y=aucs, type="o", xlab="Number of Components", ylab="AUC", pch=19)
  print("")
  print(paste("\nBest Number of Components:", best_num_components))
  print(paste("Max AUC:", max_auc))
  best_num_components
}

find_best_num_epochs = function(train_data, test_data, predictors, epochs, units=256, dropout=T)
{
  aucs = matrix(1:(3*length(epochs)), nrow=length(epochs), dimnames = list(epochs))
  print(aucs)
  i = 1
  for (epoch in epochs)
  {
    for (j in 1:3)
    {
      if (dropout)
      {
        dropout_model <- 
          keras_model_sequential() %>%
          layer_dense(units = units, activation = "relu", input_shape = ncol(train_data[,predictors]), kernel_constraint=constraint_maxnorm(3)) %>%
          layer_dropout(0.2) %>%
          layer_dense(units = units, activation = "relu", kernel_constraint=constraint_maxnorm(3)) %>%
          layer_dropout(0.1) %>%
          layer_dense(units = 1, activation = "sigmoid")  
      }
      else{
        dropout_model <- 
          keras_model_sequential() %>%
          layer_dense(units = units, activation = "relu", input_shape = ncol(train.csv[,predictors]),  kernel_regularizer = regularizer_l2(l = 0.001)) %>%
          #layer_dropout(0.0) %>%
          layer_dense(units = units, activation = "relu",  kernel_regularizer = regularizer_l2(l = 0.001)) %>%
          #layer_dropout(0.9) %>%
          layer_dense(units = 1, activation = "sigmoid")
      }
      
      
      dropout_model %>% compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = list("accuracy")
      )
      print(summary(dropout_model))
      dropout_history <- dropout_model %>% fit(
        data.matrix(train_data[,predictors]),
        data.matrix(train_data$Survived),
        epochs = epoch,
        batch_size = 32,
        #validation_data = list(data.matrix(test.csv[,predictors]), data.matrix(test.csv$Survived)),
        validation_split = 0.2,
        verbose = 2
      )
      dropout_model.pred = predict(dropout_model, data.matrix(test_data[,predictors]))
      predicted.class = ifelse(dropout_model.pred>0.5, 1, 0)
      print(table(predict=predicted.class, truth=test_data$Survived))
      
      aucs[i,j] = plotROC(dropout_model.pred, test_data)
    }
    i = i + 1
  }
  aucs
}


# Load the training and testing data
setwd("D:\\_Code\\R\\Titanic-Machine-Learning")
train.csv=read.csv("data\\train.csv", header=T)
test.csv.kaggle=read.csv("data\\test.csv", header=T)


# Data preprocessing
#train.csv$Survived = sapply(train.csv$Survived, as.factor)
train.csv$Title = sapply(train.csv$Title, as.numeric)
train.csv$Sex = ifelse(train.csv$Sex == "male", 1, 0)
train.csv$Embarked = sapply(train.csv$Embarked, as.numeric)
train.csv$Age[is.na(train.csv$Age)] = median(c(train.csv$Age, test.csv.kaggle$Age), na.rm=TRUE)

#test.csv.kaggle$Survived = sapply(test.csv.kaggle$Survived, as.factor)
test.csv.kaggle$Title = sapply(test.csv.kaggle$Title, as.numeric)
test.csv.kaggle$Sex = ifelse(test.csv.kaggle$Sex == "male", 1, 0)
test.csv.kaggle$Embarked = sapply(test.csv.kaggle$Embarked, as.numeric)
test.csv.kaggle$Age[is.na(test.csv.kaggle$Age)] = median(c(train.csv$Age, test.csv.kaggle$Age), na.rm=TRUE)
test.csv.kaggle$Fare[is.na(test.csv.kaggle$Fare)] = median(c(train.csv$Fare, test.csv.kaggle$Fare), na.rm=TRUE)
test.csv.kaggle.passenger_id = test.csv.kaggle$PassengerId


Pclass.FamSize = train.csv[,"Pclass"] * train.csv[, "FamSize"]
train.csv = cbind(train.csv, Pclass.FamSize)
Pclass.FamSize = test.csv.kaggle[, "Pclass"] * test.csv.kaggle[, "FamSize"]
test.csv.kaggle = cbind(test.csv.kaggle, Pclass.FamSize)

Title.Pclass = train.csv[,"Title"] * train.csv[, "Pclass"]
train.csv = cbind(train.csv, Title.Pclass)
Title.Pclass = test.csv.kaggle[, "Title"] * test.csv.kaggle[, "Pclass"]
test.csv.kaggle = cbind(test.csv.kaggle, Title.Pclass)
 
Title.FamSize = train.csv[,"Title"] * train.csv[, "FamSize"]
train.csv = cbind(train.csv, Title.FamSize)
Title.FamSize = test.csv.kaggle[, "Title"] * test.csv.kaggle[, "FamSize"]
test.csv.kaggle = cbind(test.csv.kaggle, Title.FamSize)

# Sex.Embarked = train.csv[,"Sex"] * train.csv[, "Embarked"]
# train.csv = cbind(train.csv, Sex.Embarked)
# Sex.Embarked = test.csv.kaggle[, "Title"] * test.csv.kaggle[, "Embarked"]
# test.csv.kaggle = cbind(test.csv.kaggle, Sex.Embarked)
# 
FamSize.Embarked = train.csv[,"FamSize"] * train.csv[, "Embarked"]
train.csv = cbind(train.csv, FamSize.Embarked)
FamSize.Embarked = test.csv.kaggle[, "FamSize"] * test.csv.kaggle[, "Embarked"]
test.csv.kaggle = cbind(test.csv.kaggle, FamSize.Embarked)


Age.Pclass = train.csv[,"Age"] * train.csv[, "Pclass"]
train.csv = cbind(train.csv, Age.Pclass)
Age.Pclass = test.csv.kaggle[, "Age"] * test.csv.kaggle[, "Pclass"]
test.csv.kaggle = cbind(test.csv.kaggle, Age.Pclass)


Log_Fare = log(train.csv[,"Fare"])
Log_Fare[is.infinite(Log_Fare)] = 0
hist(Log_Fare)
train.csv = cbind(train.csv, data.frame(Log_Fare))

Log_Fare = log(test.csv.kaggle[,"Fare"])
Log_Fare[is.infinite(Log_Fare)] = 0
hist(Log_Fare)
test.csv.kaggle = cbind(test.csv.kaggle, data.frame(Log_Fare))


# Find collinearity
omitted_col = c("PassengerId", "Ticket", "Name", "Cabin")
corrplot(cor(train.csv[,!names(train.csv) %in% omitted_col]), method="circle")
omitted_col = append(omitted_col, c("FamSize", "Pclass.FamSize", "SibSp", "Fare","Title"))
plot(train.csv[,!names(train.csv) %in% omitted_col], col=colors[train.csv$Survived+1])

#corrplot(cor(train.csv[,!names(train.csv) %in% omitted_col]), method="circle")
omitted_col = append(omitted_col, c("Survived"))
predictors = names(train.csv)[!names(train.csv) %in% omitted_col]
#plot(train.csv[,predictors], col=colors[train.csv$Survived+1])
predictors = c("Title", "Pclass", "FamSize", "Age", "Sex", "Embarked", "Title.Pclass", "Title.FamSize", "FamSize.Embarked", "Age.Pclass")
highlyCorrelated = findCorrelation(cor(train.csv[,predictors]), cutoff=0.75)
predictors[highlyCorrelated]
predictors = predictors[-highlyCorrelated]


train_trans = preProcess(train.csv[,predictors], method="scale")
train.csv[,predictors] = predict(train_trans, train.csv[,predictors])

test_trans = preProcess(test.csv.kaggle[,predictors], method="scale")
test.csv.kaggle[,predictors] = predict(test_trans, test.csv.kaggle[,predictors])

control = rfeControl(functions=caretFuncs, method="cv", number=10)

results = rfe(train.csv[,predictors], train.csv$Survived, sizes=c(3:length(predictors)), rfeControl=control, method="svmRadial")
results
predictors(results)
plot(results, type=c("g", "o"))

predictors = predictors(results)



# Histogram
par(mfrow=c(3,2))
for (i in 1:length(predictors)){
  hist(train.csv[,predictors[i]], 
       main = predictors[i],
       xlab="")
}

## PCA
num_test_rows = nrow(test.csv.kaggle)
num_train_rows = nrow(train.csv)

all.data = rbind(train.csv[,predictors], test.csv.kaggle[,predictors]) 
pca = prcomp(all.data, scale=T)
colors = c("green", "red")

par(mfrow=c(1,1))
biplot(pca, scale=0, col=c("orange", "black"))
pr.var=pca$sdev ^2
pve = pr.var/sum(pr.var)
par(mfrow=c(1,2))
# Data Visualization
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1), type='b')
plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained ", ylim=c(0,1), type='b')

plot(pca$x[,1]^2, pca$x[,1],col=colors[train.csv$Survived+1])
plot((pca$x[,1]+10)^0.5, (pca$x[,2]+10),col=colors[train.csv$Survived+1])

train.csv.pca1 = data.frame(x=cbind(pca$x[1:num_train_rows,1]^2, pca$x[1:num_train_rows,1]), Survived=train.csv$Survived)
train.csv.pca2 = data.frame(x=cbind((pca$x[1:num_train_rows,1]+10)^0.5, pca$x[1:num_train_rows,2]), Survived=train.csv$Survived)

test.csv.pca2.kaggle = data.frame(x=cbind(pca$x[num_train_rows+1:num_test_rows,1]^0.5, pca$x[num_train_rows+1:num_test_rows,2]))

train.csv.pca3 = data.frame(x=pca$x[1:num_train_rows,], Survived=train.csv$Survived)
test.csv.pca3.kaggle = data.frame(x=pca$x[num_train_rows+1:num_test_rows,])

# Perform train & test split
percentage_test = 0.2
num_test_samples = nrow(train.csv) * percentage_test

test_rows = sample(nrow(train.csv), num_test_samples)
test.csv = train.csv[test_rows, ]
train.csv = train.csv[-test_rows, ]

test.csv.pca1 = train.csv.pca1[test_rows, ]
train.csv.pca1 = train.csv.pca1[-test_rows, ]

test.csv.pca2 = train.csv.pca2[test_rows, ]
train.csv.pca2 = train.csv.pca2[-test_rows, ]

test.csv.pca3 = train.csv.pca3[test_rows, ]
train.csv.pca3 = train.csv.pca3[-test_rows, ]
predictors

## Fit Support Vector Machine (SVM)
#svmfit = svm(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked + Pclass, data=train.csv, kernel ="linear", cost=0.01, scale=FALSE)

#svmfit2 = svm(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked, data=train.csv, kernel ="radial", cost=0.01, gamma=1)
#summary(svmfit)

# Find best regular SVM
#bestmod = find_best_svm(train.csv[, c("Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived")], "linear")
#summary(bestmod)
#bestmod2 = find_best_svm(train.csv[, c("Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Pclass", "Survived")], "radial")
#summary(bestmod2)


temp = append(predictors, "Survived")
bestmod = find_best_svm(train.csv[,temp], "linear")
svm.bestmod.pred = make_prediction(bestmod, test.csv)
plotROC(svm.bestmod.pred, test.csv)


bestmod2 = find_best_svm(train.csv[,temp])
svm.bestmod.pred = make_prediction(bestmod2, test.csv)
plotROC(svm.bestmod.pred, test.csv)


summary(bestmod2)

#Find best PCA model using potentially all the components
num_comp = find_best_num_components(train.csv.pca3, test.csv.pca3)
bestmod_pca = find_best_svm(train.csv.pca3[,c(1:num_comp, ncol(train.csv.pca3))])

#Find best using transformed first component
bestmod_pca1 = find_best_svm(train.csv.pca1, "linear")

# Find best using transformed first two components
bestmod_pca2 = find_best_svm(train.csv.pca2, "radial")

# Fit Neural Network
nn_fit = neuralnet(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked + Pclass, data=train.csv, hidden=3, linear.output = FALSE)
plot(nn_fit)
nn_fit2 = neuralnet(Survived ~ Sex + Age + SibSp + Parch + Fare + Embarked + Pclass, data=train.csv, hidden=5, linear.output = FALSE)
plot(nn_fit2, rep="best")

nn_fit3 = neuralnet(train.csv$Survived ~ ., data=train.csv[,!names(train.csv) %in% omitted_col], hidden=5, linear.output = FALSE)
plot(nn_fit3)

# Fit Neural Network with PCA
nn_fit_pca= neuralnet(Survived ~ ., data=train.csv.pca3[,c(1:6, ncol(train.csv.pca3))], hidden=5, linear.output = FALSE)
plot(nn_fit_pca)
# 5, 3: 0.8012179, 0.7842949
#3:  0.7801923, 0.7809615
#4: 0.8157051, 0.8060897
#5: 0.7787821, 0.7925
#6: 0.7923718, 0.8046795
#7: 0.7813462

# Fit Neural Network with Regularization
#epochs = c(50, 100, 150, 200, 250, 300)
#epochs = c(50, 100, 150, 200)
epochs = c(50, 100, 150, 200, 300)
units = 128
aucs = find_best_num_epochs(train.csv, test.csv, predictors, epochs, units)
aucs_pca = find_best_num_epochs(train.csv.pca3, test.csv.pca3, c(1:(ncol(train.csv.pca3)-5)), epochs, units)
plot(x=epochs, y=rowMeans(aucs), type="o", pch=19)
print(aucs)

print("PCA")
plot(x=epochs, y=rowMeans(aucs_pca), type="o", pch=19)
print(aucs_pca)

dropout_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu", input_shape = ncol(train.csv[,predictors]), kernel_constraint=constraint_maxnorm(3)) %>%
  layer_dropout(0.6) %>%
  layer_dense(units = 128, activation = "relu", kernel_constraint=constraint_maxnorm(3)) %>%
  layer_dropout(0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")

dropout_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

dropout_history <- dropout_model %>% fit(
  data.matrix(train.csv[,predictors]),
  data.matrix(train.csv$Survived),
  epochs = 150,
  batch_size = 32,
  #validation_data = list(data.matrix(test.csv[,predictors]), data.matrix(test.csv$Survived)),
  validation_split = 0,
  verbose = 2
)

par(mfrow=c(1,1))
summary(dropout_model)
plot(dropout_history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="orange", type="l")
lines(dropout_history$metrics$val_loss, col="skyblue")
legend("topright", c("Training","Testing"), col=c("orange", "skyblue"), lty=c(1,1))
x = evaluate(dropout_model, data.matrix(test.csv[,predictors]), data.matrix(test.csv$Survived))
predict(dropout_model, data.matrix(test.csv[,predictors]))



## Cross Evaluation
svm.pred = make_prediction(svmfit, test.csv)
svm2.pred = make_prediction(svmfit2, test.csv)
names(train.csv) %in% names(test.csv)

svm.bestmod.pred = make_prediction(bestmod, test.csv)
svm.bestmod2.pred = make_prediction(bestmod2, test.csv)
svm.bestmod3.pred = make_prediction(bestmod3, test.csv)

svm.bestmod_pca.pred = make_prediction(bestmod_pca, test.csv.pca3)
svm.bestmod_pca1.pred = make_prediction(bestmod_pca1, test.csv.pca1)
svm.bestmod_pca2.pred = make_prediction(bestmod_pca2, test.csv.pca2)

nn_fit.pred = make_prediction(nn_fit, test.csv)
nn_fit2.pred = make_prediction(nn_fit2, test.csv)
nn_fit3.pred = make_prediction(nn_fit3, test.csv)

nn_fit_pca.pred = make_prediction(nn_fit_pca, test.csv.pca3)

dropout_model.pred = predict(dropout_model, data.matrix(test.csv[,predictors]))
predicted.class = ifelse(dropout_model.pred>0.5, 1, 0)
table(predict=predicted.class, truth=test.csv$Survived)



## ROC PREDICTIONS
plotROC(svm.pred, test.csv)
plotROC(svm2.pred, test.csv)
plotROC(svm.bestmod.pred, test.csv)
plotROC(svm.bestmod2.pred, test.csv)
plotROC(svm.bestmod_pca.pred, test.csv.pca3)
plotROC(svm.bestmod_pca1.pred, test.csv.pca1)
plotROC(svm.bestmod_pca2.pred, test.csv.pca2)

plotROC(nn_fit.pred, test.csv)
plotROC(nn_fit2.pred, test.csv)
plotROC(nn_fit3.pred, test.csv)

plotROC(dropout_model.pred, test.csv)

plotROC(nn_fit_pca.pred, test.csv.pca3)

## Kaggle Submission
# Regular SVM
svm.pred = predict(bestmod2, test.csv.kaggle)
svm.pred.rounded = round(svm.pred)
svm.pred.rounded[svm.pred.rounded<0] = 0
predictors

submission = data.frame(PassengerId=test.csv.kaggle$PassengerId,Survived=svm.pred.rounded)
write.csv(submission, "svm_submission.csv", row.names=FALSE)

# pca svm
#svm.pred = predict(bestmod_pca, test.csv.pca3.kaggle)
#svm.pred.rounded = round(svm.pred)
#svm.pred.rounded[svm.pred.rounded<0] = 0
#submission = data.frame(PassengerId=test.csv.kaggle$PassengerId,Survived=svm.pred.rounded)
#write.csv(submission, "svm_pca_submission.csv", row.names=FALSE)

# NN
nn.pred = compute(nn_fit2, train.csv)
predicted.class = ifelse(nn.pred$net.result>0.5, 1, 0)
table(predict=predicted.class, truth=train.csv$Survived)

nn.pred = compute(nn_fit2, test.csv.kaggle)
predicted.class = ifelse(nn.pred$net.result>0.5, 1, 0)
submission = data.frame(PassengerId=test.csv.kaggle.passenger_id,Survived=predicted.class)
write.csv(submission, "nn_prev_submission.csv", row.names=FALSE)

# Dropout NN
nn.pred = predict(dropout_model, data.matrix(train.csv[,predictors]))
predicted.class = ifelse(nn.pred>0.5, 1, 0)
table(predict=predicted.class, truth=train.csv$Survived)

#plotROC(dropout_model, train_data)

#        truth
# predict   0   1
#       0 499  93
#       1  50 249
#        truth
# predict   0   1
#       0 502  95
#       1  47 247
# predict   0   1
#       0 506  96
#       1  43 246
#        truth
# predict   0   1
#       0 502  93
#       1  47 249
nn.pred = predict(dropout_model, data.matrix(test.csv.kaggle[,predictors]))
predicted.class = ifelse(nn.pred>0.5, 1, 0)
submission = data.frame(PassengerId=test.csv.kaggle.passenger_id,Survived=predicted.class)
write.csv(submission, "nn_submission.csv", row.names=FALSE)
