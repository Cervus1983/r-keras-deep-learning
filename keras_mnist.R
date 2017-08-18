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
		dim = c(16, 12)
	)
)


# define, compile, fit & evaluate sequential model ----
model <- keras_model_sequential() %>% 
	layer_dense(
		units = 512,
		activation = "relu",
		input_shape = dim(train_x)[2]
	) %>% 
	layer_dense(
		units = 512,
		activation = "relu"
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

model %>% fit(
	train_x,
	train_y,
	epochs = 5,
	callbacks = callback_tensorboard(),
	validation_split = .1
)

model %>% evaluate(test_x, test_y)


# first 192 incorrectly classified images ----
prediction <- apply(predict(model, test_x), 1, which.max) - 1
actual <- apply(test_y, 1, which.max) - 1
to_plot <- which(prediction != actual)[1:(16 * 12)]

images_and_labels(
	data = test_x[to_plot, ],
	labels = array(
		sprintf("%s (%s)", prediction[to_plot], actual[to_plot]),
		dim = c(16, 12)
	)
)
