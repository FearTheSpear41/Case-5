# Read Packages #

library(keras)
library(tensorflow)
library(AppliedPredictiveModeling)

# Read Data #
data(hepatic)

data <- bio
str(data)

# Change to Matrix #
data <- as.matrix(data)
dimnames(data) <- NULL

# Normalize #
data[, 1:183] <- normalize(data[, 1:183])
data[,183] <- as.numeric(data[,183]) -1
summary(data)

# Data Partition #
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.8, 0.2))
training <- data[ind==1, 1:183]
test <- data[ind==2, 1:183]
trainingtarget <- data[ind==1, 184]
testtarget <- data[ind==2, 184]

# One Hot Encoding #
trainLabels <- to_categorical(trainingtarget)
testLabels <- to_categorical(testtarget)
print(testLabels)

# Create sequential model #
model <- keras_model_sequential()
model %>%
  layer_dense(units=5, activation = 'relu', input_shape = c(183)) %>%
  layer_dense(units = 6, activation = 'softmax')
summary(model)

# Compile #
model %>%
  compile(loss = 'categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

# Fit Model #
history <- model %>%
  fit(training,
      trainLabels,
      epoch = 200,
      batch_size = 32,
      validation_split = 0.2)
plot(history)

# Evaluate model with test data #
model1 <- model %>%
  evaluate(test, testLabels)

#Prediction & Confusion matrix - test data #
prob <- model %>%
  predict_proba(test)

pred <- model %>%
  predict_classes(test)
table1 <- table(Predicted = pred, Actual = testtarget)

cbind(prob, pred, testtarget)

# Fine-tune model #
table1
model1