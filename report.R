#' ---
#' output:
#'   html_document:
#'     toc: true
#'     theme: united
#'     highlight: textmate
#' author: "Kamil Ciemniewski <kamil@ciemniew.ski>"
#' title: "My Report"
#' ---

#+ load-libraries, messages=FALSE
library(tidyverse)
library(tidymodels)

#+ download-data, messages=FALSE, warning=FALSE, cache=TRUE
fraining.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
training.path <- "pml-training.csv"

testing.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testing.path <- "pml-testing.csv"

if(!file.exists(training.path)) {
  download.file(training.url, training.path)
}

if(!file.exists(testing.path)) {
  download.file(testing.url, testing.path)
}

#+ load-data, messages=FALSE, warning=FALSE, cache=TRUE, results='hide'
training.data <- read_csv(training.path)
testing.data <- read_csv(testing.path)

add_window_id <- function(tbl) {
  tbl %>%
    group_by(user_name) %>%
    mutate(window_id = cumsum(new_window == "yes")) %>%
    ungroup()
}

add_user_window_id <- function(tbl) {
  tbl %>%
    mutate(user_window_id=paste(user_name, window_id))
}

training.data <- training.data %>%
  add_window_id() %>%
  add_user_window_id()

testing.data <- testing.data %>%
  add_window_id() %>%
  add_user_window_id()

group_vfold_cv(training.data, group = "user_window_id", v=10)
