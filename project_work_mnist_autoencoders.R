# Inspiration from bradleyboehmke.github.io/HOML/autoencoders.html

# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for data visualization

# Modeling packages
library(h2o)      # for fitting autoencoders

# Image packages
library(jpeg)

# import the MNIST data set
mnist <- dslabs::read_mnist()
names(mnist)
### [1] "train" "test"

# initialize h2o
h2o.no_progress() # turn off progress bar, which can slow down relatively trivial example problems
h2o.init(max_mem_size = "4G")  # initialize H2O instance

# extract a test set of images for later
test_N <- 5
ae_index <- sample(1:nrow(mnist$test$images), test_N)
ae_sampled_digits <- mnist$test$images[ae_index, ]
## preview the sample test images
par(mfrow = c(1, 3), mar=c(1, 1, 1, 1))
layout(matrix(seq_len(nrow(ae_sampled_digits)), test_N, 1, byrow = FALSE))
for(i in seq_len(nrow(ae_sampled_digits))) {
  image(matrix(ae_sampled_digits[i, ], 28, 28)[, 28:1], xaxt="n", yaxt="n")
}
## convert sample images to h2o
ae_test_images <- as.h2o(ae_sampled_digits)

visualize_results <- function(ae_combine) {
  par(mfrow = c(1, 3), mar=c(1, 1, 1, 1))
  layout(matrix(seq_len(nrow(ae_combine)), test_N, seq_len(nrow(ae_combine))/3, byrow = FALSE))
  for(i in seq_len(nrow(ae_combine))) {
    image(matrix(ae_combine[i, ], 28, 28)[, 28:1], xaxt="n", yaxt="n")
  }
}

# Convert mnist images to an h2o input data set (features)
features <- as.h2o(mnist$train$images)
dim(features)
### [1] 60000   784

# Train an autoencoder
ae1 <- h2o.deeplearning(
  x = seq_along(features),     # limiting inputs just to the images (no labels)
  training_frame = features,   # actual data input
  autoencoder = TRUE,          # Autoencoder
  hidden = 2,                  # size of coded layer
  activation = 'Tanh',         # hyperbolic tangent activation
  sparse = TRUE,               # MNIST data is VERY sparse
  ignore_const_cols = FALSE
)
## Without setting `ignore_const_cols=FALSE`, this gives warnings about
##    why certain columns get removed in the autoencoder
## Dropping those columns doesn't feel good and causes issues in reconstructing the input

# Extract the deep features
ae1_codings <- h2o.deepfeatures(ae1, features, layer = 1)
dim(ae1_codings)
### [1] 60000   2

## Whoa! We went from 784 features down to 2
## But is that the optimal solution? (hint: No)

## Let's take a look at how it does
ae1_model <- h2o.getModel(ae1@model_id)
ae1_result <- predict(ae_model, ae_test_images)

ae1_combine <- rbind(ae_sampled_digits,
                    as.matrix(ae_result))
visualize_results(ae1_combine)
## 

## Let's try searching for the optimal model

# Hyperparameter search grid
hyper_grid <- list(hidden = list(
  c(400),
  c(250),
  c(100),
  c(50),
  c(10)
))

# Execute grid search
ae_grid <- h2o.grid(
  algorithm = 'deeplearning',
  x = seq_along(features),
  training_frame = features,
  grid_id = 'autoencoder_grid',
  autoencoder = TRUE,
  activation = 'Tanh',
  hyper_params = hyper_grid,
  sparse = TRUE,
  ignore_const_cols = FALSE,
  seed = 96
)

# Print grid details
h2o.getGrid('autoencoder_grid', sort_by = 'mse', decreasing = FALSE)

## We can also visualize this nicely...
## Run the test images through our models
model_400_feat <- h2o.getModel(ae_grid@model_ids[[1]])
result_400_feat <- predict(model_400_feat, ae_test_images)

model_250_feat <- h2o.getModel(ae_grid@model_ids[[2]])
result_250_feat <- predict(model_250_feat, ae_test_images)

model_100_feat <- h2o.getModel(ae_grid@model_ids[[3]])
result_100_feat <- predict(model_100_feat, ae_test_images)

model_50_feat <- h2o.getModel(ae_grid@model_ids[[4]])
result_50_feat <- predict(model_50_feat, ae_test_images)

model_10_feat <- h2o.getModel(ae_grid@model_ids[[5]])
result_10_feat <- predict(model_10_feat, ae_test_images)

combine <- rbind(ae_sampled_digits, 
                 as.matrix(result_400_feat),
                 as.matrix(result_250_feat),
                 as.matrix(result_100_feat),
                 as.matrix(result_50_feat),
                 as.matrix(result_10_feat))

## Plot the results
par(mfrow = c(1, 3), mar=c(1, 1, 1, 1))
layout(matrix(seq_len(nrow(combine)), test_N, 6, byrow = FALSE))
for(i in seq_len(nrow(combine))) {
  image(matrix(combine[i, ], 28, 28)[, 28:1], xaxt="n", yaxt="n")
}


# Train an autoencoder
ae2 <- h2o.deeplearning(
  x = seq_along(features),     # limiting inputs just to the images (no labels)
  training_frame = features,   # actual data input
  autoencoder = TRUE,          # Autoencoder
  hidden = 100,                # size of coded layer
  activation = 'Tanh',         # hyperbolic tangent activation
  sparse = TRUE,               # MNIST data is VERY sparse
  ignore_const_cols = FALSE
)
ae2_model <- h2o.getModel(ae2@model_id)
ae2_result <- predict(ae2_model, ae_test_images)

ae2_combine <- rbind(ae_sampled_digits,
                     as.matrix(ae2_result))
visualize_results(ae2_combine)


# Use relu activation function
ae_relu <- h2o.deeplearning(
  x = seq_along(features),     # limiting inputs just to the images (no labels)
  training_frame = features,   # actual data input
  autoencoder = TRUE,          # Autoencoder
  hidden = 100,                # size of coded layer
  activation = 'Rectifier',
  sparse = TRUE,               # MNIST data is VERY sparse
  ignore_const_cols = FALSE,
  stopping_metric = "MSE"
)
ae_relu_model <- h2o.getModel(ae_relu@model_id)
ae_relu_result <- predict(ae_relu_model, ae_test_images)

ae_relu_combine <- rbind(ae_sampled_digits,
                     as.matrix(ae_relu_result))
visualize_results(ae_relu_combine)


# Try more hidden layers
ae_multiple_hidden_layers <- h2o.deeplearning(
  x = seq_along(features),     # limiting inputs just to the images (no labels)
  training_frame = features,   # actual data input
  autoencoder = TRUE,          # Autoencoder
  hidden = c(250, 100, 250), # size of hidden layers
  activation = 'Tanh',         # hyperbolic tangent activation
  sparse = TRUE,               # MNIST data is VERY sparse
  ignore_const_cols = FALSE
)
ae_multiple_hidden_layers_model <- h2o.getModel(ae_multiple_hidden_layers@model_id)
ae_multiple_hidden_layers_result <- predict(ae_multiple_hidden_layers_model, ae_test_images)

ae_multiple_hidden_layers_combine <- rbind(ae_sampled_digits,
                         as.matrix(ae_multiple_hidden_layers_result))
visualize_results(ae_multiple_hidden_layers_combine)

## Shutdown H2O instance
##h2o.shutdown