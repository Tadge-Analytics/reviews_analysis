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


  
  randomForest::randomForest(
      x = as.matrix(dtm_tfidf_weighted),
      y = factor(reviews$sentiment),
      ntree = 500
    )
