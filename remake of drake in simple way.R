
###################################################################

# load packages

library(tidyverse)
library(httr)
library(tidytext)
library(tidymodels)
library(textrecipes)
library(vip)


###################################################################

# download the files 

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip"
download.file(url, destfile = "download.zip")
unzip("download.zip", exdir = "data")


# prepare the data

prepared_files_data <- 
  dir("data", recursive = T, full.names = T, pattern = "*.txt") %>% 
  tibble(file = .) %>% 
  filter(!str_detect(file, "readme.txt")) %>% 
  mutate(data = map(file, ~read_lines(.x))) %>% 
  unnest(data) %>% 
  mutate(sentiment = as.numeric(str_sub(data, -1)),
         sentiment = if_else(sentiment == 1, "good", "bad") %>% as.factor(),
         review = str_sub(data, 1, nchar(data)-1) %>% trimws(),
         source = word(file, -1, sep = "/"),
         source = word(source, 1, sep = "_"),
         review_id = row_number()) %>% 
  select(-data, -file)


# take a look if you want
# prepared_files_data %>% 
#   unnest_tokens(word, review)




###################################################################

# let's do some modelling

# create training and evaluation splits

set.seed(123)
review_split <- initial_split(prepared_files_data, strata = sentiment)

review_train <- training(review_split)
review_test <- testing(review_split)


# set the recipe

review_rec <- review_train %>% 
  recipe(sentiment ~ review, data = .) %>% 
  step_tokenize(review) %>% 
  step_stopwords(review) %>% 
  step_tokenfilter(review, max_tokens = 500) %>% 
  step_tfidf(review) %>% 
  step_normalize(all_predictors())


# take a look at what this recipe does
# with bake() or juice()

# review_rec %>%
#   prep() %>% # works out what to do
#   # juice() %>% # to see how training dataset looks like
#   bake(new_data = review_test) # applies the required steps to the test data


###################################################################


# we are going to use glmnet's logistic regressions to determine good/bad sentiment
# we would like to use the best hyperparameter for this model
# so we now need to figure this out...

# there is only one we have to worry about:
# penalty


# specify the model wit hte tune() argument where our parameter would go

lasso_spec <- logistic_reg(penalty = tune(),
                           mixture = 1) %>%
  set_engine("glmnet")


# we are going to need to validate, which of our parameter grids works best
# so we set up for some cross-valiodations
# this time using bootstraps

set.seed(123)
review_folds <- bootstraps(review_train, strata = sentiment)





# we now create a simple grid (because we only have one parameter)
# with a set number of levels for this specific parameter -with a range provided by penalty()


lambda_grid <- grid_regular(penalty(), levels = 10)


# # how to make the grid more easily?
# lasso_spec %>% 
#   parameters() %>% 
#   grid_max_entropy(size = 10)



# we are going to run a tune_grid
# to determine how effective each grid level is at our predicting our data (assesssed with the cross-validation)


# at this point, we can specifiy a workflow, if we want...
# 


lasso_wf <- workflow() %>%
  add_recipe(review_rec) %>%
  add_model(lasso_spec)




# and use it in the tune grid

library(future)
plan(multiprocess, workers = 2)
# maybe this works better than the mac version?
# doParallel::registerDoParallel()



set.seed(123)
tictoc::tic()

lasso_grid <- tune_grid(
  lasso_wf,
  resamples = review_folds,
  grid = lambda_grid,
  metrics = metric_set(roc_auc, ppv, npv), 
  control = control_grid(verbose = TRUE)
)

tictoc::toc()
# takes 1 minute(s)


# # or just run the tune grid with the model_spec and recipe, by themselves
# 
# lasso_grid <- tune_grid(lasso_spec,
#                         review_rec,
#                         resamples = review_folds,
#                         grid = lambda_grid,
#                         metrics = metric_set(roc_auc, ppv, npv))



# in my view, this is where you would just save the lasso_grid, 
# as it contains the outcomes of our parameter search


lasso_grid

###################################################################


# take a look at how the different grid versions of the model did

lasso_grid %>%
  collect_metrics()

# pick the best model

best_auc <- lasso_grid %>%
  select_best("roc_auc")


final_lasso_wf <- finalize_workflow(lasso_wf, best_auc)



# if we did the model + recipe version of tune_grid() then we will use the below: to establish a workflow (used later, I guess)

# final_model_sepc <- finalize_model(lasso_spec, best_auc)
# 
# final_lasso_wf <- workflow() %>%
#   add_recipe(review_rec) %>%
#   add_model(final_model_sepc)




###################################################################

# or we could just enter it in manually
# we can just enter it in from here....

best_auc <- tibble(penalty = 0.0088862)

lasso_spec <- logistic_reg(penalty = best_auc$penalty,
                           mixture = 1) %>%
  set_engine("glmnet")


final_lasso_wf <- workflow() %>% 
  add_recipe(review_rec) %>% 
  add_model(lasso_spec)


###################################################################

# train this model over our entire training data
tictoc::tic()

training_fit <- final_lasso_wf %>% 
  fit(review_train) 

predict(training_fit, new_data = review_test, type = "prob") %>% 
  bind_cols(review_test) %>% 
  roc_auc(truth = sentiment, .pred_good)

tictoc::toc()





# if we wanted to extract only the model
new_ <- extract_model(training_fit)




predict(training_fit, new_data = review_test, type = "class") %>% 
  bind_cols(review_test) %>% 
  conf_mat(sentiment, .pred_class)


# but this seems to do them both in one
# and also allows us to see a summary of how this model worked



review_final <- final_lasso_wf %>% 
  last_fit(review_split)

review_final %>%
  collect_metrics()

review_final %>%
  collect_predictions() %>%
  conf_mat(sentiment, .pred_class)


review_final %>%
  collect_metrics()


review_final %>%
  collect_predictions() %>%
  roc_curve(truth = sentiment, .pred_good) %>% 
  autoplot()


  
  
  




# this sort of repeated process
# gives us an even better estimation of what our actual test data predictions are going to be

fits_over_folds <- final_lasso_wf %>% 
  fit_resamples(review_folds) 



fits_over_folds %>%
  collect_metrics() 
  


###################################################################

# we might hten be curious about which words had the biggest impact.



training_fit %>% 
  pull_workflow_fit() %>% 
  vi() %>%
  group_by(Sign) %>%
  top_n(20, wt = abs(Importance)) %>%
  ungroup() %>%
  mutate(Importance = abs(Importance),
         Variable = word(Variable, -1, sep = "_"),
         Variable = fct_reorder(Variable, Importance)) %>%
  ggplot(aes(x = Importance, y = Variable, fill = Sign)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Sign, scales = "free_y") +
  labs(y = NULL)





###################################################################


# let see how it goes on some new unseen data

new_data_to_be_scored <- tibble::tribble(
  ~"review", ~"sentiment",
  "I'm still infatuated with this phone.", "good",
  "Strike 2, who wants to be rushed.", "bad",
  "I enjoyed reading this book to my children when they were little.", "good",
  "We had a group of 70+ when we claimed we would only have 40 and they handled us beautifully.", "good",
  "The story is lame, not interesting and NEVER really explains the sinister origins of the puppets", "bad",
  "Better than you'd expect.", "good",
  "It was a huge awkward 1.5lb piece of cow that was 3/4ths gristle and fat.", "bad",
  "Yes, it's that bad.", "bad",
  "I did not expect this to be so good!", "good",
  "The only redeeming quality of the restaurant was that it was very inexpensive.", "good"
) %>% 
  mutate(sentiment = as.factor(sentiment))





full_training_fit <- final_lasso_wf %>% 
  fit(prepared_files_data) 

new_data_to_be_scored %>% 
  bind_cols(
    predict(full_training_fit, new_data = new_data_to_be_scored, type = "class"),
    predict(full_training_fit, new_data = new_data_to_be_scored, type = "prob")
  )


