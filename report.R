#' ---
#' output:
#'   html_document:
#'     toc: true
#'     theme: united
#'     highlight: textmate
#' author: "Kamil Ciemniewski <kamil@ciemniew.ski>"
#' title: "Classifying the dumbell activity correctness"
#' ---

#+ load-libraries, message=FALSE, warning=FALSE
library(tidyverse)
library(tidymodels)
library(magrittr)
library(dlookr)
library(knitr)
library(kableExtra)
library(furrr)
library(purrr)
library(glue)

plan(multiprocess)

#+ download-data, message=FALSE, warning=FALSE, cache=TRUE, collapse=TRUE
training_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training_csv"
training_path <- "pml-training.csv"

testing_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testing_path <- "pml-testing.csv"

if(!file.exists(training_path)) {
  download.file(training_url, training_path)
}

if(!file.exists(testing_path)) {
  download.file(testing_url, testing_path)
}

#+ load-data, message=FALSE, warning=FALSE, cache=TRUE, results='hide', collapse=TRUE
training_data <- read_csv(training_path)
testing_data <- read_csv(testing_path) %>% mutate(classe=NA) %>% select(-problem_id)

#+ correct-data-types, collapse=TRUE, warning=FALSE
numeric_columns <- names(
  training_data %>%
    select(-classe, -raw_timestamp_part_1,
           -raw_timestamp_part_2, -cvtd_timestamp,
           -num_window, -new_window, -user_name
    )
)

training_data[, numeric_columns] %<>% lapply(function(x) as.numeric(as.character(x)))
testing_data[, numeric_columns] %<>% lapply(function(x) as.numeric(as.character(x)))

#' ## Data Exploration

#+ classe-histogram, warning=FALSE, message=FALSE
training_data %>%
  ggplot(aes(x=classe)) +
  geom_histogram(stat='count')

#' From this we can see that the `classe` output variable is slightly imbalanced. We might
#' want to do a special kind of sampling when training.

#+ missing-vars, collapse=TRUE
missing_variables <- diagnose(training_data) %>%
  filter(missing_count > 0) %>%
  select(variables, missing_percent)

missing_variable_names <- missing_variables$variables

missing_variables %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) %>%
  scroll_box(width = "100%", height = "400px")

#' Let's now create the preprocessing recipy:

rec <- recipe(classe ~ ., data=training_data)

#' As the next step in the recipy, let's remove the variables we feel do not contribute
#' to the outcome in any way:

rec %<>% step_rm(new_window, user_name, X1)

#' We also don't want the timestamps in the dataset as the time-window data has already been
#' captured. The variables already contain the information about the rate of change of
#' different measurements within the close window of time. We're **not** dealing with time-series
#' forecasting then. As the same step, we'll also remove the variables with huge amounts
#' of missing values. As we've seen, all of them are above the 90% of rate so there's no
#' way to meaningfully impute them:

rec %<>% step_rm(contains('timestamp'), one_of(missing_variable_names))

#' We also don't want variables with "near-zero variance" as they aren't likely to contribute
#' to our outcome:

rec %<>% step_nzv(all_predictors())

#' Because we're going to use PCA we need a way to make sure the relationships between variables
#' are as linear as possible. We'll do the YeoJohnson transformation then:

rec %<>% step_YeoJohnson(all_predictors())

#' Also for the same reason we need the variables to be expressed in the same scale:

rec %<>% step_center(all_predictors())
rec %<>% step_scale(all_predictors())

#' As the last step, we'll make the pipeline turn the dataset into a matrix of principle components
#' that explains the 90% of the original variance:

rec %<>% step_pca(all_predictors(), threshold=0.9)

#' Let's see now how many principal components will survive for the model:

rec %>%
  prep %>%
  bake(new_data=training_data) %>%
  sample_n(size=100) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) %>%
  scroll_box(width = "100%", height = "400px")

mod_formula <- as.formula(classe ~ .)

#' # Defining the model stack

model_specs <- list(
  boost_tree_1 = list(
    final=FALSE,
    fun=boost_tree,
    engine = "xgboost",
    params = list(
      mtry=10
    )
  ),
  boost_tree_2 = list(
    final=FALSE,
    fun=boost_tree,
    engine = "xgboost",
    params = list(
      mtry=15
    )
  ),
  mlp_1 = list(
    final=FALSE,
    fun=mlp,
    engine = "nnet",
    params = list(
      hidden_units=16,
      dropout=0.1,
      activation='softmax'
    )
  ),
  mlp_2 = list(
    final=FALSE,
    fun=mlp,
    engine = "nnet",
    params = list(
      hidden_units=32,
      dropout=0.3,
      activation='softmax'
    )
  ),
  nearest_neighbor_1 = list(
    final=FALSE,
    fun=nearest_neighbor,
    engine = "kknn",
    params = list(
      neighbors=3
    )
  ),
  nearest_neighbor_2 = list(
    final=FALSE,
    fun=nearest_neighbor,
    engine = "kknn",
    params = list(
      neighbors=5
    )
  ),
  svm_rbf_1 = list(
    final=FALSE,
    fun=svm_rbf,
    engine = "kernlab",
    params = list(
      rbf_sigma=0.1
    )
  ),
  svm_rbf_2 = list(
    final=FALSE,
    fun=svm_rbf,
    engine = "kernlab",
    params = list(
      rbf_sigma=0.2
    )
  ),
  final_model_spec = list(
    final=TRUE,
    fun=boost_tree,
    engine = "xgboost",
    params = list(
      mtry=15
    )
  )
)

#' Defining the final prediction model that will take the predictions of other
#' models into account to make it a stacked ensemble:


train <- function(dataset) {
  # return the tibble with trained models
}

construct_models <- function(model_specs) {
  model_from_spec <- function(model_spec) {
    args <- rlang::duplicate(model_spec$params)
    args$mode = "classification"
    do.call(model_spec$fun, args)
  }

  tibble(
    name=names(model_specs),
    model=map(model_specs, model_from_spec),
    final=map_int(model_specs, function(m) m$final)
  )
}

predict_classes <- function(dataset, mod_formula) {
  # return tibble with each model result
}

predict_results <- function(dataset, mod_formula) {
  # return tibble with each model result
}

cv_folds <- vfold_cv(training_data, strata="classe", v=10, repeats=10)

cv_folds %<>% mutate(recipes = map(splits, prepper, recipe = rec, retain = TRUE))

cv_folds %<>% mutate(results=map(splits, predict_results, mod_formula))

cv_folds %<>% mutate(results=map(splits, predict_results, mod_formula))

#' Let's do our final submission predictions:

trained_models <- train(training_data)
predicted <- predict_classes(training_data, trained_models)
