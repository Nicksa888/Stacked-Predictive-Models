#########################
#########################
#### Clear Workspace ####
#########################
#########################

rm(list = ls()) 
# clear global environment to remove all loaded data sets, functions and so on.

###################
###################
#### Libraries ####
###################
###################

library(easypackages) # enables the libraries function
suppressPackageStartupMessages(
  libraries("rsample", # creating train-test data splits
            "recipes", # for minor feature engineering tasks
            "h2o", # for fitting stacked models
            "caret",
            "purrr"
  ))

###############################
###############################
#### Set Working Directory ####
###############################
###############################

setwd("C:/R Portfolio/Stacked Models/Data")

bikes <- read.csv("bikes.csv")
str(bikes)
glimpse(bikes)
summary(bikes)

# Convert categorical variables into factors

bikes$season <- as.factor(bikes$season)
bikes$holiday <- as.factor(bikes$holiday)
bikes$weekday <- as.factor(bikes$weekday)
bikes$weather <- as.factor(bikes$weather)

# Convert numeric variables into integers

bikes$temperature <- as.integer(bikes$temperature)
bikes$realfeel <- as.integer(bikes$realfeel)
bikes$windspeed <- as.integer(bikes$windspeed)

levels(bikes$season) <- c("Spring", "Summer", "Autumn", "Winter")

# remove column named date
bikes <- bikes %>% select(-date)
str(bikes)

###############################
###############################
# Training and Test Data Sets #
###############################
###############################

set.seed(1234) # changing this alters the make up of the data set, which affects predictive outputs

split <- initial_split(bikes, 
                       strata = "rentals")

bikes_train <- training(split)
bikes_test <- testing(split)

# Ensure we have consistent categorical levels #

blueprint <- recipe(rentals ~ ., 
                    bikes_train) %>%
  step_other(all_nominal(), threshold = 0.005)

# Create training and test data sets for h2o

Sys.setenv(JAVA_HOME = "C:/Program Files/Java/jdk-11.0.12") # your own path of Java SE installed
h2o.init()

# Train set

train_h2o <- prep(blueprint, 
                  training = bikes_train, 
                  retain = T) %>%
                  juice() %>%
                  as.h2o()
  
# Test Set 

test_h2o <- prep(blueprint, 
                  training = bikes_train) %>%
  bake(new_data = bikes_test) %>%
  as.h2o()

# Set response and feature names 

Y <- "rentals"
X <- setdiff(names(bikes_train), Y)

##########################
##########################
# Create a Stacked Model #
##########################
##########################

# Train and validate a GLM Model

best_glm <- h2o.glm(
  x = X,
  y = Y,
  training_frame = train_h2o,
  alpha = 0.1,
  remove_collinear_columns = T, 
  nfolds = 10,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = T,
  seed = 123)

# Train and cross validate a Random Forest Model

best_rf <- h2o.randomForest(
  x = X,
  y = Y,
  training_frame = train_h2o,
  ntrees = 1000, 
  mtries = -1,
  max_depth = 30, 
  min_rows = 1,
  sample_rate = 0.8,
  nfolds = 10,
  score_each_iteration = TRUE,
  score_tree_interval = 0,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = T,
  seed = 123,
  stopping_rounds = 50,
  stopping_metric = "RMSE",
  stopping_tolerance = 0.001
  )
?h2o.randomForest
# Train and cross validate a GBM Model 

best_gbm <- h2o.gbm(
  x = X, 
  y = Y,
  training_frame = train_h2o,
  ntrees = 5000,
  learn_rate = 0.01,
  max_depth = 7,
  min_rows = 5,
  sample_rate = 0.8,
  nfolds = 10,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = T,
  seed = 123,
  stopping_rounds = 50,
  stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# Train and Cross Validate a XGBoost Model

h2o.xgboost.available() # check if it is available. Not available on Windows machines

?h2o.xgboost
best_xgb <- h2o.xgboost(
  x = X, 
  y = Y,
  training_frame = train_h2o,
  ntrees = 5000,
  learn_rate = 0.05,
  max_depth = 3,
  min_rows = 3,
  sample_rate = 0.8,
  categorical_encoding = "Enum",
  nfolds = 10,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = T,
  seed = 123,
  stopping_rounds = 50,
  stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# Train a stacked tree ensemble 

ensemble_tree <- h2o.stackedEnsemble(
  x = X,
  y = Y,
  training_frame = train_h2o,
  model_id = "my_tree_ensemble",
  base_models = list(best_glm, best_rf, best_gbm),
  metalearner_algorithm = "drf"
  )

# Get results from base learners 

get_rmse <- function(model) {
  results <- h2o.performance(model, 
                             newdata = test_h2o)
                             results@metrics$RMSE
}

list(best_glm, best_rf, best_gbm) %>%
  purrr::map_dbl(get_rmse)

# Stacked Results

h2o.performance(ensemble_tree, newdata = test_h2o)@metrics$RMSE

# The stacked ensemble performs better than all models apart from the random forest

# Check correlation among base learner predictions 

glm_id <- best_glm@model$cross_validation_holdout_predictions_frame_id
rf_id <- best_rf@model$cross_validation_holdout_predictions_frame_id
gbm_id <- best_gbm@model$cross_validation_holdout_predictions_frame_id

data.frame(
  GLM_pred = as.vector(h2o.getFrame(glm_id$name)),
  RF_pred = as.vector(h2o.getFrame(rf_id$name)),
  GBM_pred = as.vector(h2o.getFrame(gbm_id$name))
  ) %>% cor()

# The base learners all have very high correlations with each other, so stacking provides less advantage in this situation, especially as the RMSE in the stacked model was
# worse than in the Random Forest Model

##########################
##########################
# Stacking a Grid Search #
##########################
##########################

# An alternative approach involves stacking multiple models from the same base learner
# Through stacking the results of a grid search, it is possible to capitalise on the benefits of each model in the grid search to produce a meta model.
# The following performs a random search across a wide range of GBM hyerparameter settings and stops after 25 models have been run

# Define GBM hyperparameter grid

hyper_grid <- list(
  max_depth = c(1, 3, 5),
  min_rows = c(1, 5, 10),
  learn_rate = c(0.01, 0.05, 0.1),
  learn_rate_annealing = c(0.99, 1),
  sample_rate = c(0.5, 0.75, 1),
  col_sample_rate = c(0.8, 0.9, 1)
)

# Define Random Grid Search Criteria 

search_criteria <- list(
  strategy = "RandomDiscrete",
  max_models = 25
  )

# Build Random Grid Search

random_grid <- h2o.grid(
  algorithm = "gbm",
  grid_id = "gbm_grid",
  x = X, 
  y = Y,
  training_frame = train_h2o,
  hyper_params = hyper_grid,
  search_criteria = search_criteria,
  ntrees = 5000, 
  stopping_metric = "RMSE",
  stopping_rounds = 10,
  stopping_tolerance = 0,
  nfolds = 10,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = T,
  seed = 123
)

# Sort Results by RMSE

h2o.getGrid(
  grid_id = "gbm_grid",
  sort_by = "rmse"
)

# Grab the model id for the top model, chosen by validation error

best_model_id <- random_grid@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)
h2o.performance(best_model, newdata = test_h2o)

# Train a stacked ensemble using the GBM grid

ensemble <- h2o.stackedEnsemble(
  x = X, 
  y = Y,
  training_frame = train_h2o,
  model_id = "ensemble_gbm_grid",
  base_models = random_grid@model_ids, 
  metalearner_algorithm = "gbm"
)

# Evaluate Performance on the Test Set

h2o.performance(ensemble, newdata = test_h2o)

# The RMSE value is lower for the stacked grid model rather than the ensemble based on the stacked grid models and is therefore a model which reflects the data better

##############################
##############################
# Automated Machine Learning #
##############################
##############################

# This involves performing an automated search across multiple base learners and stack the resulting models
# Here, instead of searching across a variety of parameters in a single base learner, we search across a variety of hyperparameter settings for many different base learners

?h2o::h2o.automl # for all the supported models

# Use AutoML to identify a list of candidate models
auto_ml <- h2o.automl(
  x = X, 
  y = Y,
  training_frame = train_h2o,
  nfolds = 10,
  max_runtime_secs = 60 * 120,
  max_models = 50,
  keep_cross_validation_predictions = T,
  sort_metric = "RMSE",
  stopping_rounds = 50, 
  stopping_metric = "RMSE",
  stopping_tolerance = 0,
  seed = 123
)

# Assess the model leaderboard

auto_ml@leaderboard %>%
  as.data.frame() %>%
  dplyr::select(model_id, rmse) %>%
  dplyr::slice(1:25)

# Grab the model id for the top auto model, chosen by validation error

best_auto_model_id <- auto_ml@leaderboard[[1]]
best_model <- best_auto_model_id
h2o.performance(best_model, newdata = test_h2o)
