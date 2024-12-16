# Load Libraries
install.packages("caret")
install.packages("e1071")
install.packages("ggplot2")
install.packages("lattice")
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(ggplot2)

# Load the dataset
diabetes_data <- read.csv(file.choose())

# Normalize features
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
diabetes_data[ ,1:8] <- lapply(diabetes_data[ ,1:8], normalize)

# Convert target to factor for classification
diabetes_data$Outcome <- as.factor(diabetes_data$Outcome)

# Train-test split
set.seed(123)
trainIndex <- createDataPartition(diabetes_data$Outcome, p = 0.8, list = FALSE)
train_data <- diabetes_data[trainIndex, ]
test_data <- diabetes_data[-trainIndex, ]

# Store accuracies for comparison
accuracy_results <- data.frame(Model = character(), Accuracy = numeric())

# 1. Linear Regression
lm_model <- lm(as.numeric(Outcome) ~ ., data = train_data)
lm_predictions <- predict(lm_model, test_data)
lm_class <- as.factor(ifelse(lm_predictions > 0.5, 1, 0))
cm_lm <- confusionMatrix(lm_class, test_data$Outcome)
accuracy_lm <- cm_lm$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "Linear Regression", Accuracy = accuracy_lm))

# Plot: Actual vs Predicted
ggplot(data.frame(Actual = as.numeric(test_data$Outcome), Predicted = lm_predictions), aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Linear Regression: Actual vs Predicted", x = "Actual", y = "Predicted") +
  theme_minimal()

# 2. Logistic Regression
log_model <- glm(Outcome ~ ., data = train_data, family = "binomial")
log_prob <- predict(log_model, test_data, type = "response")
log_class <- as.factor(ifelse(log_prob > 0.5, 1, 0))
cm_log <- confusionMatrix(log_class, test_data$Outcome)
accuracy_log <- cm_log$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "Logistic Regression", Accuracy = accuracy_log))

# Plot: Predicted probabilities by class
ggplot(data.frame(Actual = test_data$Outcome, Predicted_Prob = log_prob), aes(x = Actual, y = Predicted_Prob)) +
  geom_boxplot(aes(fill = Actual)) +
  labs(title = "Logistic Regression: Predicted Probabilities", x = "Actual Outcome", y = "Predicted Probability") +
  theme_minimal()

# 3. Naive Bayes
nb_model <- naiveBayes(Outcome ~ ., data = train_data)
nb_predictions <- predict(nb_model, test_data)
cm_nb <- confusionMatrix(nb_predictions, test_data$Outcome)
accuracy_nb <- cm_nb$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "Naive Bayes", Accuracy = accuracy_nb))

# Plot: Predicted probabilities by class
nb_prob <- predict(nb_model, test_data, type = "raw")
nb_prob_df <- data.frame(Actual = test_data$Outcome, Prob_Class_0 = nb_prob[, 1], Prob_Class_1 = nb_prob[, 2])
ggplot(nb_prob_df, aes(x = Actual)) +
  geom_boxplot(aes(y = Prob_Class_0, fill = "Class 0"), alpha = 0.5) +
  geom_boxplot(aes(y = Prob_Class_1, fill = "Class 1"), alpha = 0.5) +
  labs(title = "Naive Bayes: Predicted Probabilities", x = "Actual Outcome", y = "Predicted Probability", fill = "Class") +
  theme_minimal()

# 4. Support Vector Machine (SVM)
svm_model <- svm(Outcome ~ ., data = train_data, kernel = "linear")
svm_predictions <- predict(svm_model, test_data)
cm_svm <- confusionMatrix(svm_predictions, test_data$Outcome)
accuracy_svm <- cm_svm$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "SVM", Accuracy = accuracy_svm))

# Plot: Decision boundary (using 2 features)
xrange <- seq(min(test_data$Glucose), max(test_data$Glucose), length.out = 100)
yrange <- seq(min(test_data$BMI), max(test_data$BMI), length.out = 100)
grid <- expand.grid(Glucose = xrange, BMI = yrange)
grid$Predicted <- predict(svm_model, newdata = data.frame(grid))
ggplot(grid, aes(x = Glucose, y = BMI, fill = Predicted)) +
  geom_tile(alpha = 0.5) +
  geom_point(data = test_data, aes(x = Glucose, y = BMI, color = Outcome), size = 2) +
  labs(title = "SVM: Decision Boundary", x = "Glucose", y = "BMI", fill = "Predicted", color = "Actual") +
  theme_minimal()

# 5. Decision Tree
tree_model <- rpart(Outcome ~ ., data = train_data, method = "class")
tree_predictions <- predict(tree_model, test_data, type = "class")
cm_tree <- confusionMatrix(tree_predictions, test_data$Outcome)
accuracy_tree <- cm_tree$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "Decision Tree", Accuracy = accuracy_tree))

# Plot: Decision Tree
rpart.plot(tree_model, main = "Decision Tree")

# Final Accuracy Plot
ggplot(accuracy_results, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = round(Accuracy, 2)), vjust = -0.5, size = 4) +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal()

