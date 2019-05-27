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

document_seed <- 1986

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
training_data <- read_csv(training_path) %>% mutate(classe=as.factor(classe))
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

set.seed(document_seed)

rec %>%
  prep %>%
  bake(new_data=training_data) %>%
  sample_n(size=100) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) %>%
  scroll_box(width = "100%", height = "400px")

#' # Defining the model stack

model_specs <- list(
  boost_tree_1 = list(
    final=FALSE,
    fun=boost_tree,
    engine = "xgboost",
    params = list(
      tree_depth=9
    )
  ),
  rand_forest_1 = list(
    final=FALSE,
    fun=rand_forest,
    engine = "randomForest",
    params = list(
      trees=128
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
  svm_rbf_1 = list(
    final=FALSE,
    fun=svm_rbf,
    engine = "kernlab",
    params = list(
      rbf_sigma=0.1
    )
  ),
  final_model_spec = list(
    final=TRUE,
    fun=boost_tree,
    engine = "xgboost",
    params = list(
      tree_depth=9
    )
  )
)

#' Defining the final prediction model that will take the predictions of other
#' models into account to make it a stacked ensemble:

construct_models <- function() {
  model_from_spec <- function(model_spec) {
    args <- rlang::duplicate(model_spec$params)
    args$mode = "classification"
    do.call(model_spec$fun, args)
  }

  tibble(
    name=names(model_specs),
    model=map(model_specs, model_from_spec),
    model_spec=model_specs,
    final=map_int(model_specs, function(m) m$final)
  )
}

predict_results <- function(model, model_spec, analysis_data, assessment_data, final_assessment_data) {
  set.seed(document_seed)

  model %<>%
    set_engine(model_spec$engine) %>%
    fit(classe ~ ., data=analysis_data)

  list(
    analysis_results=predict(model, new_data=assessment_data) %>% pull(.pred_class),
    assessment_results=predict(model, new_data=final_assessment_data) %>% pull(.pred_class)
  )
}

predict_final_results <- function(model, model_spec, analysis_data, assessment_data) {
  set.seed(document_seed)

  model %>%
    set_engine(model_spec$engine) %>%
    fit(classe ~ ., data=analysis_data) %>%
    predict(new_data=assessment_data) %>%
    pull(.pred_class)
}

predict_all_results <- function(analysis_data, assessment_data, models) {
  # appends predictions to each model row in models

  basic_models <- models %>% filter(final == 0)
  final_model <- models %>% filter(final == 1)

  base_final_split <- initial_split(analysis_data, prop=0.5)
  base_analysis_data <- analysis(base_final_split)
  final_analysis_data <- assessment(base_final_split)

  model_predictions <- future_map2(
    basic_models$model,
    basic_models$model_spec,
    predict_results,
    base_analysis_data,
    final_analysis_data,
    assessment_data
  )

  analysis_results <- model_predictions %>%
    map(function(preds){ tibble(classe=preds$analysis_results) }) %>%
    bind_cols

  assessment_results <- model_predictions %>%
    map(function(preds){ tibble(classe=preds$assessment_results) }) %>%
    bind_cols

  names(analysis_results) <- basic_models$name
  names(assessment_results) <- basic_models$name

  final_analysis_data <- bind_cols(final_analysis_data, analysis_results)
  final_assessment_data <- bind_cols(assessment_data, assessment_results)

  final_results <- predict_final_results(
    final_model$model[[1]],
    final_model$model_spec[[1]],
    final_analysis_data,
    final_assessment_data
  )

  assessment_results %<>% mutate(final_results=final_results)

  models$predictions <- map(names(assessment_results), function(name) { assessment_results[[name]] })

  models$metric <- map(models$predictions, function(preds) {
    metrics(tibble(preds=preds, truth=assessment_data$classe), truth, preds)
  })

  models
}

predict_all_recipy_results <- function(recipy, models) {
  predict_all_results(
    bake(recipy$parameters, new_data=analysis(recipy$data_split)),
    bake(recipy$parameters, new_data=assessment(recipy$data_split)),
    models
  )
}

split_prepper <- function(data_split, recipy) {
  list(
    parameters=prep(recipy, analysis(data_split)),
    data_split=data_split
  )
}

if(file.exists('cv_folds.rda')) {
  load(file='data.rda')
}

cv_folds <- vfold_cv(training_data, strata="classe", v=10, repeats=10)

cv_folds %<>% mutate(recipes=future_map(splits, split_prepper, recipy = rec))

cv_folds %<>% mutate(models=map(splits, function(split)  construct_models()))

if(!exists('accuracies')) {
  cv_folds %<>% mutate(models=map2(recipes, models, predict_all_recipy_results))

  accuracies <- map_dbl(1:length(model_specs), function(i) {
    mean(map_dbl(cv_folds$models, function(model) { model$metric[[i]][[3]][[1]]}))
  })

  names(accuracies) <- names(model_specs)

  save(cv_folds, accuracies, file='data.rds')
}

accuracies %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))

#' Let's do our final submission predictions:

