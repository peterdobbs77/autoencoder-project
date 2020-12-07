# Inspiration from bradleyboehmke.github.io/HOML/autoencoders.html
# Data from https://stanfordmlgroup.github.io/competitions/chexpert/

# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for data visualization

# Image packages
library(jpeg)

# import CheXpert data
base_path <- "C:/Users/Peter/Documents"
training_summary <- read.csv(file.path(base_path, "CheXpert-v1.0-small/train.csv"))
train_frontal <- training_summary[training_summary$Frontal.Lateral == "Frontal", ]$Path

# # Investigate image size constraints
# image.sizes <- array(numeric(),c(0,2))
# for(entry in train_frontal){
#   image_i <- readJPEG(file.path(base_path, entry))
#   image.sizes <- rbind(image.sizes, dim(image_i))
# }

## For Normalizing Images
M <- 320
N <- 390

# image_i <- readJPEG(file.path(base_path, train_frontal[1]))
# if(dim(image_i)[1] < M) {
#   image_i <- rbind(image_i, matrix(0, nrow=(M-dim(image_i)[1]), ncol=dim(image_i)[2]))
# }
# if(dim(image_i)[2] < N) {
#   image_i <- cbind(image_i, matrix(0, nrow=M, ncol=(N-dim(image_i)[2])))
# }
nFeatures <- M*N
train_combine <- array(numeric(), c(0, nFeatures))
for (entry in train_frontal) {
  image_i <- readJPEG(file.path(base_path, entry))
  if(dim(image_i)[1] != M || dim(image_i)[2] != N) {
    next
  }
  train_combine <- rbind(train_combine, array(as.vector(image_i), c(1, nFeatures)))
}

# Modeling packages
library(h2o)      # for fitting autoencoders
h2o.init(max_mem_size = "15g")  # initialize H2O instance

train_frontal_features <- as.h2o(train_combine)
dim(train_frontal_features)
### [1] ??   124800

# Train an autoencoder
frontal_ae_simple <- h2o.deeplearning(
  x = seq_along(train_frontal_features),     # limiting inputs just to the images (no labels)
  training_frame = train_frontal_features,   # actual data input
  autoencoder = TRUE,          # Autoencoder
  hidden = nFeatures/2,                  # size of coded layer
  activation = 'Tanh',         # hyperbolic tangent activation
  sparse = TRUE,               # MNIST data is VERY sparse
  ignore_const_cols = FALSE
)