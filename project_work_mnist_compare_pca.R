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


# initialize h2o
h2o.no_progress() # turn off progress bar, which can slow down relatively trivial example problems
h2o.init(max_mem_size = "4G")  # initialize H2O instance

## convert sample images to h2o
ae_test_images <- as.h2o(ae_sampled_digits)

# Convert mnist images to an h2o input data set (features)
features <- as.h2o(mnist$train$images)
dim(features)
### [1] 60000   784

# Perform Principal Component Analysis
pca <- h2o.prcomp(
  training_frame = features,
  pca_method = "GramSVD",
  k = ncol(features),
  ignore_const_cols = FALSE,
  transform = "STANDARDIZE",
  impute_missing = TRUE,
  max_runtime_secs = 1000
)

pca@model$eigenvectors %>% 
  as.data.frame() %>% 
  mutate(feature = row.names(.)) %>%
  ggplot(aes(pc1, pc2, label = feature)) +
  geom_text()

# Compute eigenvalues
eigen <- pca@model$importance["Standard deviation", ] %>%
  as.vector() %>%
  .^2
# Find PCs where the sum of eigenvalues is greater than or equal to 1
subset <- which(eigen >= 1)
## First 164 PCs

array_eigenvalues <- as.numeric(eigen)[subset]
par(mfrow=c(1,2))
layout(matrix(seq_len(2), 1, 2))
plot(array_eigenvalues/sum(array_eigenvalues),
     xlab="Principal Component",
     ylab="Variance Explained",
     ylim=c(0,1)
) + abline(h=0.9)
plot(cumsum(array_eigenvalues)/sum(array_eigenvalues),
     xlab="Principal Component",
     ylab="Cumulative Variance Explained",
     ylim=c(0,1)
) + abline(h=0.9)

eigenvectors <- pca@model$eigenvectors[subset]
matrix_eigenvectors <- matrix(numeric(), c(length(eigenvectors[,1]),0))
for(vector in eigenvectors){
  matrix_eigenvectors <- cbind(matrix_eigenvectors, as.vector(vector))
}

pca_model <- h2o.getModel(pca@model_id)
pca_result <- h2o.predict(pca_model, newdata = ae_test_images)
pca_result_matrix <- as.matrix(pca_result[subset])
pca_reconstructed <- pca_result_matrix %*% t(matrix_eigenvectors)
pca_combined <- rbind(ae_sampled_digits,
                      pca_reconstructed)

## Plot the results
par(mfrow = c(test_N, 2), mar=c(1, 1, 1, 1))
layout(matrix(seq_len(test_N*2), test_N, 2, byrow = FALSE))
for(i in seq_len(nrow(pca_combined))) {
  image(matrix(pca_combined[i, ], 28, 28)[, 28:1], xaxt="n", yaxt="n")
}

## Shutdown H2O instance
# # h2o.shutdown()
