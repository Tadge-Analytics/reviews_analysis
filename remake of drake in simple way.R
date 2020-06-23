library(tidyverse)


# # download the files
# library(httr)
# url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip"
# download.file(url, destfile = "download.zip")
# unzip("download.zip", exdir = "data")


prepared_files_data <- 
  dir("data", recursive = T, full.names = T, pattern = "*.txt") %>% 
  tibble(file = .) %>% 
  filter(!str_detect(file, "readme.txt")) %>% 
  mutate(data = map(file, ~read_lines(.x))) %>% 
  unnest(data) %>% 
  mutate(result = as.numeric(str_sub(data, -1)),
         review = str_sub(data, 1, nchar(data)-1),
         source = word(file, -1, sep = "/"),
         source = word(source, 1, sep = "_"),
         review_id = row_number()) %>% 
  select(-data, -file)



###################################################################

processed_documents <- prepared_files_data %>% 
  pull(review) %>% 
  tolower %>%
  tm::removeNumbers() %>%
  tm::removePunctuation() %>% 
  text2vec::word_tokenizer() %>% 
  SnowballC::wordStem(language = "en") 
    




  token_iterator <- text2vec::itoken(
    processed_documents,
    progressbar = FALSE
  )

  vocabulary <- text2vec::create_vocabulary(
    token_iterator,
    stopwords = tidytext::stop_words$word
  )

  vocabulary <- text2vec::prune_vocabulary(
    vocabulary,
    doc_proportion_min = 0,
    doc_proportion_max = 1
  )



###################################################################

  
# training hte model

randomForest::randomForest(
      x = as.matrix(dtm_tfidf_weighted),
      y = factor(reviews$sentiment),
      ntree = 500
    )



  
  ###################################################################


# some new data

new_data_to_be_scored <- function() {
  tibble::tribble(
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
  )
}





###################################################################

# this is some check that the model is good

# validate_model <- function(random_forest, vectoriser, tfidf = NULL) {
#   model_sentiment <- function(x) sentiment(x, random_forest, vectoriser, tfidf)
#   oob <- random_forest$err.rate[random_forest$ntree, "OOB"]
# 
#   assertthat::assert_that(model_sentiment("love") == "good")
#   assertthat::assert_that(model_sentiment("bad") == "bad")
#   assertthat::assert_that(oob < 0.4)
# 
#   TRUE
# }