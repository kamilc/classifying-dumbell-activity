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

#' # Defining the model stack as the recipe step

resamples <- vfold_cv(training_data, v=10, repeats=10)


