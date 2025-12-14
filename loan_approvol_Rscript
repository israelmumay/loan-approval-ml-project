library(caret)
library(kernlab)
library(tidyverse)
library(class)
library(e1071)
library(readr)
library(pROC)
library(ggplot2)
#Load the Dataset
data <- read_csv("C:/Users/rober/OneDrive/loan_data.csv")
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
# Create dummy variables
dummies <- model.matrix(~ . - 1, data = data[categorical_vars[categorical_vars != "loan_status"]])
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
test_data <- full_data[-train_index, ]
# Separate features and target
train_x <- train_data[, names(train_data) != "loan_status"]
test_x <- test_data[, names(test_data) != "loan_status"]
train_y <- train_data$loan_status
test_y <- test_data$loan_status
#SMV
set.seed(2)
train_y <- factor(train_y)
test_y <- factor(test_y)
# Grid search for cost and gamma
cost_values <- c(0.1, 1, 10)
gamma_values <- c(0.001, 0.01, 0.1)
results <- data.frame()
for (Cval in cost_values) {
for (Gval in gamma_values) {
model <- svm(
x = train_x,
y = train_y,
kernel = "radial",
cost = Cval,
gamma = Gval,
probability = TRUE
)
preds <- predict(model, test_x)
acc <- mean(preds == test_y)
err <- 1 - acc
results <- rbind(results, data.frame(
cost = Cval,
gamma = Gval,
accuracy = acc,
error = err
))
}
}
# View grid search results
print(results)
# Select best hyperparameters (highest accuracy)
best <- results[which.max(results$accuracy), ]
best_cost <- best$cost
best_gamma <- best$gamma
print(best)
# Train SVM with best parameters
svm_best <- svm(
x = train_x,
y = train_y,
kernel = "radial",
cost = best_cost,
gamma = best_gamma,
probability = TRUE
)
test_pred_class <- predict(svm_best, test_x)
test_pred_prob <- attr(predict(svm_best, test_x, probability = TRUE), "probabilities")[, "approved"]
conf_mat <- confusionMatrix(test_pred_class, test_y, positive = "approved")
print(conf_mat)
roc_obj <- roc(test_y, test_pred_prob, levels = rev(levels(test_y)))
auc_val <- auc(roc_obj)
print(auc_val)
full_y <- full_data$loan_status
full_x <- full_data[, names(full_data) != "loan_status"]
svm_full <- svm(
x = full_x,
y = full_y,
kernel = "radial",
cost = best_cost,
gamma = best_gamma,
probability = TRUE
)
print(svm_full)
print(svm_full$tot.nSV)
# Predictions on full dataset
full_pred <- predict(svm_full, full_x)
conf_full <- confusionMatrix(full_pred, full_y)
print(conf_full)
# Select top 2 numeric variables correlated with loan_status
cor_with_target <- sapply(numeric_vars, function(x) cor(as.numeric(full_data[[x]]), as.numeric(full_data$loan_status)))
top_vars <- names(sort(abs(cor_with_target), decreasing = TRUE))[1:2]
top_vars
# Prepare plotting dataframe
plot_df <- full_data[, c(top_vars, "loan_status")]
names(plot_df)[1:2] <- c("X1","X2")
# Train SVM on 2D predictors
svm_2d <- svm(
loan_status ~ X1 + X2,
data = plot_df,
kernel = "radial",
cost = best_cost,
gamma = best_gamma,
probability = TRUE
)
# Create prediction grid
x1_range <- seq(min(plot_df$X1), max(plot_df$X1), length.out = 200)
x2_range <- seq(min(plot_df$X2), max(plot_df$X2), length.out = 200)
grid <- expand.grid(X1 = x1_range, X2 = x2_range)
grid$pred <- predict(svm_2d, grid)
#Variable importance regarding the model
set.seed(2)
svm_model_caret <- train(
loan_status ~ .,
data = train_data,
method = "svmRadial",
tuneGrid = expand.grid(C = best_cost, sigma = best_gamma),
trControl = trainControl(method = "none")
)
importance <- varImp(svm_model_caret, scale = TRUE)
print(importance)
plot(importance, top = 10)
# Plot decision boundary
ggplot() +
geom_raster(data = grid, aes(x = X1, y = X2, fill = pred), alpha = 0.25) +
geom_contour(
data = grid,
aes(x = X1, y = X2, z = as.numeric(pred)),
breaks = 1.5, # boundary between class 1 and 2
color = "black",
linewidth = 1.0
) +
scale_fill_manual(values = c("rejected" = "red", "approved" = "blue")) +
geom_point(
data = plot_df,
aes(x = X1, y = X2, color = loan_status),
size = 1.2,
alpha = 0.7
) +
scale_color_manual(values = c("rejected" = "red", "approved" = "blue")) +
labs(
title = "SVM Decision Boundary (Radial Kernel)",
x = top_vars[1],
y = top_vars[2],
fill = "Prediction",
color = "Actual"
) +
theme_minimal()
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
