# devtools::install_github("rstudio/keras")
library(keras)
library(tidyverse)


# fetch data ----
data <- dataset_mnist()

train_x <- data$train$x
train_y <- data$train$y

test_x <- data$test$x
test_y <- data$test$y

rm(data)


# flatten & normalise input ----
train_x <- array(train_x, dim = c(dim(train_x)[1], prod(dim(train_x)[-1]))) / 255
test_x <- array(test_x, dim = c(dim(test_x)[1], prod(dim(test_x)[-1]))) / 255


# one-hot encode labels ----
train_y <- to_categorical(train_y, 10)
test_y <- to_categorical(test_y, 10)


# plots data (num [1:(M * N), 1:784]) and labels (chr [1:M, 1:N]) ----
images_and_labels <- function(data, labels) {
	image_nrow <- dim(labels)[1]
	image_ncol <- dim(labels)[2]
	
	tibble(label = array(labels)) %>% 
		mutate(
			x = (row_number() - 1) %% image_ncol + 1,
			y = image_nrow - (row_number() - 1) %/% image_ncol
		) %>%
		ggplot() + 
			annotation_raster(
				raster = do.call(
						rbind,
						lapply(
							1:image_nrow,
							function(row_index) do.call(
								cbind,
								lapply(
									((row_index - 1) * image_ncol + 1):(row_index * image_ncol),
									function(image_index) 1 - array(data[image_index, ], c(28, 28))
								)
							)
						)
					),
				xmin = -Inf, xmax = Inf,
				ymin = -Inf, ymax = Inf
			) + 
			geom_text(
				aes(x = x, y = y, label = label),
				color = "red",
				nudge_x = .4, nudge_y = .3,
				size = 3
			) + 
			coord_cartesian(xlim = c(1, image_ncol), ylim = c(1, image_nrow)) + 
			theme_void()
}


# first 192 train images ----
images_and_labels(
	data = train_x[1:192, ],
	labels = array(
		apply(train_y[1:192, ], 1, which.max) - 1,
		dim = c(12, 16)
	)
)


# define, compile, fit & evaluate sequential model ----
model <- keras_model_sequential() %>% 
	layer_conv_2d(
		filters = 32,
		kernel_size = c(3, 3),
		padding = "same",
		activation = "relu",
		kernel_initializer = "he_uniform",
		kernel_regularizer = regularizer_l2(l = .0001),
		input_shape = c(28, 28, 1)
	) %>% 
	layer_batch_normalization() %>% 
	layer_conv_2d(
		filters = 32,
		kernel_size = c(3, 3),
		padding = "same",
		activation = "relu",
		kernel_initializer = "he_uniform",
		kernel_regularizer = regularizer_l2(l = .0001)
	) %>% 
	layer_batch_normalization() %>% 
	layer_max_pooling_2d(pool_size = 2) %>% 
	layer_dropout(rate = .25) %>% 
	layer_flatten() %>% 
	layer_dense(
		units = 128,
		activation = "relu",
		kernel_initializer = "he_uniform",
		kernel_regularizer = regularizer_l2(l = .0001)
	) %>% 
	layer_batch_normalization() %>% 
	layer_dropout(rate = .5) %>% 
	layer_dense(
		units = 10,
		activation = "softmax",
		kernel_regularizer = regularizer_l2(l = .0001)
	)

model %>% compile(
	optimizer = "adam",
	loss = "categorical_crossentropy",
	metrics = c("accuracy")
)

#tensorboard("logs/keras_mnist_tuning")

model %>% fit(
	array(train_x, dim = c(dim(train_x)[1], 28, 28, 1)),
	train_y,
	epochs = 50,
	callbacks = list(
		callback_early_stopping(patience = 5),
		callback_tensorboard(log_dir = "logs/keras_mnist_tuning/run_4")
	),
	validation_split = .1
)

model %>% evaluate(array(test_x, dim = c(dim(test_x)[1], 28, 28, 1)), test_y)


# incorrectly classified images ----
prediction <- apply(predict(model, array(test_x, dim = c(dim(test_x)[1], 28, 28, 1))), 1, which.max) - 1
actual <- apply(test_y, 1, which.max) - 1
to_plot <- which(prediction != actual)

images_and_labels(
	data = test_x[to_plot, ],
	labels = array(
		sprintf("%s (%s)", prediction[to_plot], actual[to_plot]),
		dim = c(7, 12)
	)
)


# data augmentation ----
datagen <- image_data_generator(
	width_shift_range = .1,
	height_shift_range = .1
)

# explicitly split training & validation sets
val_x <- tail(train_x, 6000)
val_y <- tail(train_y, 6000)

train_x <- head(train_x, -6000)
train_y <- head(train_y, -6000)

datagen %>% fit_image_data_generator(array(train_x, dim = c(dim(train_x)[1], 28, 28, 1)))

model %>% fit_generator(
	flow_images_from_data(
		array(train_x, dim = c(dim(train_x)[1], 28, 28, 1)),
		train_y,
		datagen
	),
	steps_per_epoch = dim(train_x)[1],
	epochs = 5,
	callbacks = callback_tensorboard(log_dir = "logs/keras_mnist_tuning/run_5"),
	validation_data = list(array(val_x, dim = c(dim(val_x)[1], 28, 28, 1)), val_y)
)
