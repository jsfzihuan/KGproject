# randomForest_classification_example.R
# In this script, we demonstrate how to use the randomForest package for a classification
# problem.
# Step 1. Load the data and simple exploration
# Step 2. Split the data into training and test sets
# Step 3. Benchmark: train a single decision tree using rpart
# Step 4. train 2 randome forest models using different parameters
# Step 5. Combine 3 random forest models together using the combine function
# Step 6. Continuously grow a random forest by training more trees, and
# plot the accuracy of training and test as the number of trees increases.
# Step 7. Check the importance of all variables

# Remove all objects in the workspace first
rm(list=ls())

# Check required package is installed or not. If not, install it.
# 1. randomForest
randomForest.installed <- 'randomForest' %in% rownames(installed.packages())
if (randomForest.installed) {
  print("the randomForest package is already installed, let's load it...")
}else {
  print("let's install the randomForest package first...")
  install.packages('randomForest', dependencies=T)
}
library('randomForest')
# 2. mlbench
mlbench.installed <- 'mlbench' %in% rownames(installed.packages())
if (mlbench.installed) {
  print("the mlbench package is already installed, let's load it...")
}else {
  print("let's install the mlbench package first...")
  install.packages('mlbench', dependencies=T)
}
library(mlbench)

#========================
# Step 1. Load the data and simple exploration
# load the Sonar dataset, and change the name of the last column to 'target'
data(Sonar, package='mlbench')
D <- Sonar
colnames(D)[ncol(D)] <- 'target'


# show the type for each col
for(i in 1:ncol(D)) {
  msg <- paste('col ', i, ' and its type is ', class(D[,i]))
  print(msg)
}

# Step 2. Split the data into training and test sets
# Randomly split the whole data set into a training and a test data set
# After spliting, we have the training set: (X_train, y_train)
# and the test data set: (X_test, y_test)
train_ratio <- 0.7
n_total <- nrow(D)
n_train <- round(train_ratio * n_total)
n_test <- n_total - n_train
set.seed(42)
list_train <- sample(n_total, n_train)
D_train <- D[list_train,]
D_test <- D[-list_train,]

y_train <- D_train$target
y_test <- D_test$target

# Step 3. Benchmark: train a single decision tree using rpart
library('rpart')
M_rpart1 <- rpart(target~., data = D_train)
print('show the summary of the trained model')
summary(M_rpart1)

# Compute the performance on the training and test data sets
y_test_pred_rpart1 <- predict(M_rpart1, D_test, type='class')
accuracy_test_rpart1 <- sum(y_test==y_test_pred_rpart1) / n_test
msg <- paste0('accuracy_test_rpart1 = ', accuracy_test_rpart1)
print(msg)

y_train_pred_rpart1 <- predict(M_rpart1, D_train, type='class')
accuracy_train_rpart1 <- sum(y_train==y_train_pred_rpart1) / n_train
msg <- paste0('accuracy_train_rpart1 = ', accuracy_train_rpart1)
print(msg)


# Step 4. train 2 randome forest models using different parameters
# Step 4.1 Train a random forest model using default parameters
# the default number of trees is 500
M_randomForest1 <- randomForest(target~., data = D_train)
print('show the summary of the trained model')
summary(M_randomForest1)
# We can get the confusion matrix using the print function directlty
print(M_randomForest1)
# We can check the number of trees of the trained model
ntree1 <- M_randomForest1$ntree
msg <- paste0('number of trees in M_randomForest1 = ', ntree1)
print(msg)

# Get the prediction on the test set and compute the accuracy
y_test_pred_rf1 <- predict(M_randomForest1, D_test, type='response')
accuracy_test_rf1 <- sum(y_test == y_test_pred_rf1) / n_test
msg <- paste0('accuracy_test_rf1 = ', accuracy_test_rf1)
print(msg)

y_train_pred_rf1 <- predict(M_randomForest1, D_train, type='response')
accuracy_train_rf1 <- sum(y_train == y_train_pred_rf1) / n_train
msg <- paste0('accuracy_train_rf1 = ', accuracy_train_rf1)
print(msg)

# We train a second model using fewer decision trees, and control the 
# complexity of each tree.
M_randomForest2 <- randomForest(target~., data = D_train, ntree=100, mtry=3, 
                                do.trace=10, importance=T)
print('show the summary of the trained model')
summary(M_randomForest2)
print(M_randomForest2)
# We can check the number of trees of the trained model
ntree2 <- M_randomForest2$ntree
msg <- paste0('number of trees in M_randomForest2 = ', ntree2)
print(msg)


# Get the prediction on the test set and compute the accurac
y_test_pred_rf2 <- predict(M_randomForest2, D_test, type='response')
accuracy_test_rf2 <- sum(y_test == y_test_pred_rf2) / n_test
msg <- paste0('accuracy_test_rf2 = ', accuracy_test_rf2)
print(msg)

y_train_pred_rf2 <- predict(M_randomForest2, D_train, type='response')
accuracy_train_rf2 <- sum(y_train == y_train_pred_rf2) / n_train
msg <- paste0('accuracy_train_rf2 = ', accuracy_train_rf2)
print(msg)


# Step 5. Combine 3 random forest models together using the combine function
# Train 3 random forest models
M_rf_base1 <- randomForest(target~., data = D_train, ntree = 15)
M_rf_base2 <- randomForest(target~., data = D_train, ntree = 20)
M_rf_base3 <- randomForest(target~., data = D_train, ntree = 10)
# Combine these 2 models just trained
M_rf_comb <- combine(M_rf_base1, M_rf_base2, M_rf_base3)
print(M_rf_comb)

# compute the performance on the test data set
y_test_pred_rf_base1 <- predict(M_rf_base1, D_test, type='response')
accuracy_test_rf_base1 <- sum(y_test == y_test_pred_rf_base1) / n_test
msg <- paste0('accuracy_test_rf_base1 = ', accuracy_test_rf_base1)
print(msg)

y_test_pred_rf_base2 <- predict(M_rf_base2, D_test, type='response')
accuracy_test_rf_base2 <- sum(y_test == y_test_pred_rf_base2) / n_test
msg <- paste0('accuracy_test_rf_base2 = ', accuracy_test_rf_base2)
print(msg)

y_test_pred_rf_base3 <- predict(M_rf_base3, D_test, type='response')
accuracy_test_rf_base3 <- sum(y_test == y_test_pred_rf_base3) / n_test
msg <- paste0('accuracy_test_rf_base3 = ', accuracy_test_rf_base3)
print(msg)

y_test_pred_rf_comb <- predict(M_rf_comb, D_test)
accuracy_test_rf_comb <- sum(y_test == y_test_pred_rf_comb) / n_test
msg <- paste0('accuracy_test_rf_comb = ', accuracy_test_rf_comb)
print(msg)


# Step 6. Continuously grow a random forest by training more trees, and
# plot the accuracy of training and test as the number of trees increases.

ntree_list <- 1:200
accuracy_train_list <- rep(0, length(ntree_list))
accuracy_test_list <- rep(0, length(ntree_list))
ntree_length = length(ntree_list)
for (i in 1:ntree_length) {
  # Build the random forest model based on the existing random forest model
  if (i==1) {
    M_rf_base <- randomForest(target~., data = D_train, ntree = ntree_list[1])
  }else {
    ntree_delta <- ntree_list[i] - ntree_list[i-1]
    M_rf_base <- grow(M_rf_base, ntree_delta)
  }
  # Compute accuracy on training and test data set
  y_train_pred_rfi <- predict(M_rf_base, D_train)
  y_test_pred_rfi <- predict(M_rf_base, D_test) 
  accuracy_train_rfi <- sum(y_train == y_train_pred_rfi) / n_train
  accuracy_test_rfi <- sum(y_test == y_test_pred_rfi) / n_test
  accuracy_train_list[i] <- accuracy_train_rfi
  accuracy_test_list[i] <- accuracy_test_rfi
}
# Plot the training and test accuracy
y_min = min(min(accuracy_test_list), min(accuracy_train_list)) - 0.1
y_max = max(max(accuracy_test_list), max(accuracy_train_list)) + 0.1
plot(range(ntree_list), c(y_min, y_max), type='n', xlab='ntree', ylab='accuracy')
lines(ntree_list, accuracy_train_list, type='l', lty=1, col='black')
lines(ntree_list, accuracy_test_list, type='l', lty=2, col='red')
legend_char_list <- c('training data', 'test data')
# We can use locator(1) to specify the legend position by clicking the mouse
#legend(locator(1), legend_char_list, cex=1.2, col=c('red', 'black'), lty=c(1,2))
# Or we can specify the legend position directly
legend("topright", legend_char_list, cex=1.2, col=c('red', 'black'), lty=c(1,2))

# Step 7. Check the importance of all variables
# We train a new random forest model consisting of 100 trees
M_rf_imp <- randomForest(target~., data = D_train, ntree = 100, importance = T)
print('we show the variable importance using OOB samples')
var_imp1 <- importance(M_rf_imp, type=1)
print(var_imp1)

print('we show the variable importance using tree node impurity/MSE descrease')
var_imp2 <- importance(M_rf_imp, type=2)
print(var_imp2)
