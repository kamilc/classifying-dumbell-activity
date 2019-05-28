#' ---
#' output:
#'   html_document:
#'     theme: united
#'     highlight: textmate
#' author: "Kamil Ciemniewski <kamil@ciemniew.ski>"
#' title: "Classifying the dumbell activity correctness"
#' ---

#' # Summary
#' This document presents step-by-step process of processing the fitness devices data to classify the activity into 5 groups of correctness. Here's the description of the challenge:
#'
#'
#' > Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.
#'
#'
#' # Notes for peer-reviewers
#'
#' This work uses the family of R packages called [tidymodels](https://github.com/tidymodels). It's a successor of the [caret](http://topepo.github.io/caret/index.html) package that this course was based on. It's been created by the same author. The reason was the ease of conducting croiss-validation the **proper** way as descibed in this report's body.
#'
#' Please note also that this file has been compiled out of a regular R source file - not the Rmd "R Markdown" one. This hasn't been covered in any of the specialization's courses but you can do it by later compiling the file by hand with: `rmarkdown::render('report.R')`.
#'
#' # Data exploration
#'
#' Let's start off by loading up the libraries we'll use:

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

#' We are going to use `furrr` which provides functional style functions while allowing us ot distribute the work across many cores easily. To make that parallelization happen we need to choose the correct plan:

#+ plan-multiprocess
plan(multiprocess)

#' To make things easier to debug and / or reproduce, let's choose the document seed value. We'll use it in each parallel process when working on data:

#+ set-document-seed
document_seed <- 1986

#' We need to download the data files if the haven't been downloaded yet:

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

#' Now we need to load them up into memory. I'm using `tidyverse` family of tools. Here's the code that loads up the data from the CSV files directly into `tibble` type of variables:

#+ load-data, message=FALSE, warning=FALSE, cache=TRUE, results='hide', collapse=TRUE
training_data <- read_csv(training_path) %>% mutate(classe=as.factor(classe))
testing_data <- read_csv(testing_path) %>% mutate(classe=NA) %>% select(-problem_id)

#' Notice that we've set the `classe` variable to be the factor which is very important for our future, prediction work. We're also removing the `problem_id` column as it doesn't seem to be adding any value.

#' Let's have a quick peek into the loaded data:

#+ peek-into-data
set.seed(document_seed)

training_data %>%
  glimpse()

#' All numerical data have been gathered from the devices. We only have statistics to tell us if some values appear to be outlying or not. In this work I'm choosing to treat those values as they are, trusting in device's ability to gather data correctly.

#' The data appears to contain time-series data. Upon a closer inspection though we can see that the each row already contains the time-context as the values have been averaged over the windows of time. We don't need the timestamps then and we can treat the problem as a regular classification. Additional columns that don't appear to have influence on the correctness of exersize are `num_window`, `new_window` and `user_name`. Let's get rid of them right now:

#+ correct-data-types, collapse=TRUE, warning=FALSE
numeric_columns <- names(
  training_data %>%
    select(-classe, -raw_timestamp_part_1,
           -raw_timestamp_part_2, -cvtd_timestamp,
           -num_window, -new_window, -user_name
    )
)

#' There seem to be some variables containing numerical data that were coded in strings. We need to parse those numbers correctly. Notice the use of the `%<>%` operator that comes from the `magrittr` package. It's rarely see nand the meaning is "pipe to the right but assign the result back":

#+ character-to-numeric, warning=FALSE, message=FALSE
training_data[, numeric_columns] %<>% lapply(function(x) as.numeric(as.character(x)))
testing_data[, numeric_columns] %<>% lapply(function(x) as.numeric(as.character(x)))

#' Firs thing we want to see is what is the distribution of the outcome variable:

#+ classe-histogram, warning=FALSE, message=FALSE
training_data %>%
  ggplot(aes(x=classe)) +
  geom_histogram(stat='count')

#' From this we can see that the `classe` output variable is slightly imbalanced. We might
#' want to do a special kind of sampling when training.

#' Next up, we need to see how much data in this dataset is missing:

#+ missing-vars, collapse=TRUE
missing_variables <- diagnose(training_data) %>%
  filter(missing_count > 0) %>%
  select(variables, missing_percent)

missing_variable_names <- missing_variables$variables

missing_variables %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) %>%
  scroll_box(width = "100%", height = "400px")

#' We can see that for all the columns containing missing data, the percentage is extremely high - with almost all rows having NA's. We're not going to imputate this data then. All those columns are going to be removed.

#' Now, the plan of attack is to do 10-fold cross validation - 10 times (each time with different splits to better estimate the out of sample error). Very often you can see the data processing being done on the whole trainign set, and then the cross-validation being performed. This is a huge mistake and results in the "data leakage" (more in this [here](https://machinelearningmastery.com/data-leakage-machine-learning/)). What we will do instead is to move the data preprocessing into the cross-validation, by doing it on each fold separately. The reason is that transofrmations like scaling depend on the sample mean. If you were to do it on the whole training set, you'd leak the information about this mean into each testing split.

#' A good way to move the preprocessing to be done on each split separately is to use the [recipes](https://tidymodels.github.io/recipes/) package. It allows us to define the processing pipeline and then later to use it on each split independently.

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
#' are as linear as possible. We'll do the [Yeo Johnson](https://www.stat.umn.edu/arc/yjpower.pdf) transformation then:

rec %<>% step_YeoJohnson(all_predictors())

#' Also for the same reason we need the variables to be expressed in the same scale:

rec %<>% step_center(all_predictors())
rec %<>% step_scale(all_predictors())

#' As the last step, we'll make the pipeline turn the dataset into a matrix of principle components
#' that explains the 95% of the original variance:

rec %<>% step_pca(all_predictors(), threshold=0.95)

#' Let's see now how many principal components will survive for the model:

set.seed(document_seed)

rec %>%
  prep %>%
  bake(new_data=training_data) %>%
  sample_n(size=10) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) %>%
  scroll_box(width = "100%", height = "400px")

#' # Defining the model stack

#' The plan is to use [model stacking](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/) approach to model ensembing. We're going to define the list of base models. We'll also define the final model that will receive the main data with base model's predictions bound column-wise. Here's the list of model definitions:

model_specs <- list(
  boost_tree_1 = list(
    final=FALSE,
    fun=boost_tree,
    engine = "xgboost",
    params = list(
      learn_rate=0.1,
      trees=512
    )
  ),
  rand_forest_1 = list(
    final=FALSE,
    fun=rand_forest,
    engine = "randomForest",
    params = list(
      trees=256
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
    fun=rand_forest,
    engine = "randomForest",
    params = list(
      trees=512
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

#' We'll need a function that will take the model itself, the model's spec as defined above and the datasets to work on:

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

#' There's also a need to a function that will make the final, stacked predictions:

predict_final_results <- function(model, model_spec, analysis_data, assessment_data) {
  set.seed(document_seed)

  model %>%
    set_engine(model_spec$engine) %>%
    fit(classe ~ ., data=analysis_data) %>%
    predict(new_data=assessment_data) %>%
    pull(.pred_class)
}

#' To make out lifes easier, we'll need to wrap this whole process in an easy to use function. It takes the training and testing sets as well as the `tibble` of models:

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

#' Next, we'll also need a wrapper around the main process of doing training and predictions model-wise:

predict_all_recipy_results <- function(recipy, models) {
  predict_all_results(
    bake(recipy$parameters, new_data=analysis(recipy$data_split)),
    bake(recipy$parameters, new_data=assessment(recipy$data_split)),
    models
  )
}

#' Following is another wrapper to be used with the `rsample` package we're using in this document for organizing of the cross-validation:

split_prepper <- function(data_split, recipy) {
  list(
    parameters=prep(recipy, analysis(data_split)),
    data_split=data_split
  )
}

#' Doing 10-fold cross-validation, 10 times on 5 different models is going to take some time. Let's load up results if we already have them saved:

if(file.exists('data.rda')) {
  load(file='data.rda')
}

#' If they haven't been loaded though, let's define the data split, construct the models, prepare the data preprocessing parameters and do the actualy training and testing in splits. Notice the `strata` argument to `vfold_cv` that's going to sample the splits to make the classes as balanced as possible:

if(!exists('accuracies')) {
  cv_folds <- vfold_cv(training_data, strata="classe", v=10, repeats=10)
  cv_folds %<>% mutate(recipes=future_map(splits, split_prepper, recipy = rec))
  cv_folds %<>% mutate(models=map(splits, function(split)  construct_models()))
  cv_folds %<>% mutate(models=map2(recipes, models, predict_all_recipy_results))

  accuracies <- map_dbl(1:length(model_specs), function(i) {
    mean(map_dbl(cv_folds$models, function(model) { model$metric[[i]][[3]][[1]]}))
  })

  names(accuracies) <- names(model_specs)

  save(cv_folds, accuracies, file='data.rda')
}

#' Let's see how well our models were doing on average:

accuracies %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))

#' We can see that the theory behind the model stacking is sound indeed.

#' Let's do our final submission predictions now:

models <- construct_models()

#+ main-prepping, warning=FALSE, cache=TRUE, collapse=TRUE
main_prepped <- prep(rec, training_data)

test_results <- predict_all_results(juice(main_prepped), bake(main_prepped, new_data=testing_data), models)

test_results$predictions[[5]]
