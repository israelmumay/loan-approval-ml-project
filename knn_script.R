#install.packages("caret")
library(caret)
library(tidyverse)   
library(class)        
library(e1071)        
library(readr)

#Load the Dataset
data <- read_csv("~/Desktop/loan_data.csv")

head(data)
str(data)

#Exploring the Dataset

dim(data)
names(data)
summary(data)
colSums(is.na(data))
sum(duplicated(data))
data$loan_status <- factor(data$loan_status,
                           levels = c(0, 1),
                           labels = c("rejected", "approved"))
    # Identify numeric variables
numeric_vars <- names(data)[sapply(data, is.numeric)]
numeric_vars

    # Identify categorical variables
categorical_vars <- names(data)[sapply(data, is.factor) | sapply(data, is.character)]
categorical_vars


#checking outliers

for (col in numeric_vars) {
  Q1 <- quantile(data[[col]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[col]], 0.75, na.rm = TRUE)
  IQR_val <- IQR(data[[col]], na.rm = TRUE)
  
  outliers <- sum(data[[col]] < (Q1 - 1.5*IQR_val) | 
                    data[[col]] > (Q3 + 1.5*IQR_val), na.rm = TRUE)
  
  cat(col, "- Number of outliers:", outliers, "\n")
}

#DATA cleaning

data$Loan_ID <- NULL
#handling outliers
for (col in numeric_vars) {
  p1 <- quantile(data[[col]], 0.01, na.rm = TRUE)
  p99 <- quantile(data[[col]], 0.99, na.rm = TRUE)
  
  data[[col]][data[[col]] < p1] <- p1
  data[[col]][data[[col]] > p99] <- p99
}
data$person_income <- log1p(data$person_income)
data$loan_amnt <- log1p(data$loan_amnt)
data$loan_percent_income <- log1p(data$loan_percent_income)



for (col in numeric_vars) {
  p1 <- quantile(data[[col]], 0.01)
  p99 <- quantile(data[[col]], 0.99)
  
  min_val <- min(data[[col]])
  max_val <- max(data[[col]])
  
  cat(col, 
      "- Min:", min_val, 
      "| 1st Percentile:", p1,
      "- Max:", max_val, 
      "| 99th Percentile:", p99, "\n")
}


#DATA VISUALIZATION & ANALYSIS



for (col in numeric_vars) {
  hist(data[[col]], 
       main = paste("Histogram of", col), 
       xlab = col, 
       col = "lightblue", 
       border = "white")
}

for (col in numeric_vars) {
  boxplot(data[[col]], 
          main = paste("Boxplot of", col), 
          col = "lightgreen")
}

for (col in categorical_vars) {
  barplot(table(data[[col]]),
          main = paste("Bar Plot of", col),
          col = "orange",
          las = 2)
}

for (col in numeric_vars) {
  boxplot(data[[col]] ~ data$loan_status, 
          main = paste(col, "by Loan Status"),
          xlab = "Loan Status",
          ylab = col,
          col = "lightblue")
}


cor_matrix <- cor(data[numeric_vars], use = "complete.obs")
cor_matrix
corrplot::corrplot(cor_matrix, method = "color")

# MODELING 


# Make target a factor
data$loan_status <- as.factor(data$loan_status)

# Convert categorical to factor
for (col in categorical_vars) {
  data[[col]] <- as.factor(data[[col]])
}
categorical_vars <- setdiff(categorical_vars, "loan_status")

# Create dummy variables
dummies <- model.matrix(~ . - 1, data = data[categorical_vars])
dummies <- as.data.frame(dummies)

# Scale numeric variables
scaled_numeric <- scale(data[numeric_vars])
scaled_numeric <- as.data.frame(scaled_numeric)

# Combine into final dataset
full_data <- cbind(scaled_numeric, dummies, loan_status = data$loan_status)


# TRAIN TEST SPLIT 

set.seed(2)

train_index <- sample(1:nrow(full_data), 0.8 * nrow(full_data))

train_data <- full_data[train_index, ]
test_data  <- full_data[-train_index, ]

# Separate features and target
train_x <- train_data[, names(train_data) != "loan_status"]
test_x  <- test_data[, names(test_data) != "loan_status"]

train_y <- train_data$loan_status
test_y  <- test_data$loan_status


# BASELINE KNN WITH k = 5

knn_pred <- knn(train = train_x, test = test_x, cl = train_y, k = 5)

# Accuracy
mean(knn_pred == test_y)

# Confusion matrix
table(Predicted = knn_pred, Actual = test_y)

# KNN TUNING 


set.seed(2)
control <- trainControl(method = "cv", number = 5)

knn_model <- train(
  loan_status ~ .,
  data = train_data,
  method = "knn",
  tuneLength = 30,
  trControl = control
)
knn_model$bestTune
knn_model
plot(knn_model)
best_k <- knn_model$bestTune$k

final_knn_pred <- knn(
  train = train_x,
  test = test_x,
  cl = train_y,
  k = best_k
)
final_accuracy <- mean(final_knn_pred == test_y)
final_accuracy
test_error <- 1 - final_accuracy
test_error
table(Predicted = final_knn_pred, Actual = test_y)


full_x <- full_data[, names(full_data) != "loan_status"]
full_y <- full_data$loan_status

final_full_knn <- knn(train = full_x, test = full_x, cl = full_y, k = best_k)

full_accuracy <- mean(final_full_knn == full_y)
full_accuracy

table(Predicted = final_full_knn, Actual = full_y)

colnames(full_data)
colnames(full_data) <- make.names(colnames(full_data))
install.packages("randomForest")
library(randomForest)

set.seed(2)

rf_model <- randomForest(
  loan_status ~ .,
  data = full_data,
  importance = TRUE,
  ntree = 500
)

importance(rf_model)
varImpPlot(rf_model, main = "Variable Importance for Loan Status")

