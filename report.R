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
training_data <- read_csv(training_path) %>% mutate(set="training")
testing_data <- read_csv(testing_path) %>% mutate(set="testing", classe=NA) %>% select(-problem_id)
dataset <- rbind(training_data, testing_data)

#+ correct-data-types, collapse=TRUE, warning=FALSE
numeric_columns <- names(
  dataset %>%
    select(-classe, -set, -raw_timestamp_part_1,
           -raw_timestamp_part_2, -cvtd_timestamp,
           -num_window, -new_window, -user_name
    )
)

dataset[, numeric_columns] %<>% lapply(function(x) as.numeric(as.character(x)))

#' ## Data Exploration

#+ missing-vars, collapse=TRUE
missing_variables <- diagnose(dataset) %>%
  filter(missing_count > 0 & variables != "classe") %>%
  select(variables, missing_percent)

missing_variable_names <- missing_variables$variables

missing_variables %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) %>%
  scroll_box(width = "100%", height = "400px")

#' Let's now create the preprocessing recipy:

rec <- recipe(classe ~ ., data=dataset) %>%
  step_rm(set, contains('timestamp'), one_of(missing_variable_names)) %>%
  step_nzv(all_predictors())
