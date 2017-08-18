# devtools::install_github("rstudio/keras")
library(grid)
library(keras)
library(tidyverse)


# fetch data ----
data <- dataset_cifar10()

train_x <- data$train$x
train_y <- data$train$y

test_x <- data$test$x
test_y <- data$test$y

rm(data)

# https://www.cs.toronto.edu/~kriz/cifar.html
classes <- c("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


# flatten & normalise input ----
tmp <- max(train_x)
train_x <- train_x / tmp
test_x <- test_x / tmp


# one-hot encode labels ----
tmp <- length(unique(train_y))
train_y <- to_categorical(train_y, tmp)
test_y <- to_categorical(test_y, tmp)


# visualisation function ----
fetch_image <- function(data, i) array(
	rgb(data[i, , , 1], data[i, , , 2], data[i, , , 3]),
	dim(data)[2:3]
)

#images <- function(data, indices) grid.raster(
#	do.call(
#		rbind,
#		lapply(
#			1:dim(indices)[2],
#			function(i) do.call(cbind, lapply(indices[, i], function(j) fetch_image(data, j)))
#		)
#	),
#	interpolate = FALSE
#)

images_and_labels <- function(data, indices, labels, size = 3) tibble(
	label = array(labels),
	row_num = 1:length(labels)
) %>% 
	mutate(
		x = (row_num - 1) %% dim(indices)[1],
		y = dim(indices)[2] - (row_num - 1) %/% dim(indices)[1]
	) %>% 
	ggplot() + 
		annotation_raster(
			raster = do.call(
				rbind,
				lapply(
					1:dim(indices)[2],
					function(i) do.call(cbind, lapply(1:dim(indices)[1], function(j) fetch_image(data, (i - 1) * dim(indices)[1] + j)))
				)
			),
			xmin = -Inf, xmax = Inf,
			ymin = -Inf, ymax = Inf
		) + 
		geom_text(
			aes(x = x, y = y, label = label),
			color = "red",
			hjust = "center",
			size = size
		) + 
		theme_void() + 
		scale_x_continuous(expand = c(0, 0)) + 
		scale_y_continuous(expand = c(0, 0)) + 
		coord_cartesian(xlim = c(0, dim(indices)[1]) - .5, ylim = c(.5, dim(indices)[2] + .5))


# look at training set ----
images_and_labels(
	data = train_x,
	indices = array(1:(59 * 36), dim = c(59, 36)),
	labels = rep(NA, 16 * 12)
)

images_and_labels(
	data = train_x,
	indices = array(1:(16 * 12), dim = c(16, 12)),
	labels = classes[apply(train_y[1:(16 * 12), ], 1, which.max)]
)


# define, compile, fit & evaluate sequential model ----
model <- keras_model_sequential() %>% 
	layer_conv_2d(
		filters = 32,
		kernel_size = c(3, 3),
		padding = "same",
		activation = "relu",
		input_shape = dim(train_x)[-1]
	) %>% 
	layer_conv_2d(
		filters = 32,
		kernel_size = c(3, 3),
		padding = "same",
		activation = "relu"
	) %>% 
	layer_max_pooling_2d(
		pool_size = 2
	) %>% 
	layer_dropout(
		rate = .25
	) %>% 
	layer_conv_2d(
		filters = 64,
		kernel_size = c(3, 3),
		padding = "same",
		activation = "relu"
	) %>% 
	layer_conv_2d(
		filters = 64,
		kernel_size = c(3, 3),
		padding = "same",
		activation = "relu"
	) %>% 
	layer_max_pooling_2d(
		pool_size = 2
	) %>% 
	layer_dropout(
		rate = .5
	) %>% 
	layer_flatten() %>% 
	layer_dense(
		units = 512,
		activation = "relu"
	) %>% 
	layer_dropout(
		rate = .5
	) %>% 
	layer_dense(
		units = 10,
		activation = "softmax"
	)

model %>% compile(
	optimizer = "adam",
	loss = "categorical_crossentropy",
	metrics = c("accuracy")
)

summary(model)

model %>% fit(
	train_x,
	train_y,
	epochs = 140,
	callbacks = callback_tensorboard(log_dir = "logs/cifar10/run_3"),
	validation_split = .1
)

#model %>% save_model_hdf5("cifar_150_epochs.hdf5")

model %>% evaluate(test_x, test_y)


# look at results ----
actual <- apply(test_y, 1, which.max)
prediction <- apply(predict(model, test_x), 1, which.max)

images_and_labels(
	data = test_x,
	indices = array(1:(16 * 12), dim = c(16, 12)),
	labels = classes[ifelse(actual != prediction, prediction, NA)[1:(16 * 12)]]
)
