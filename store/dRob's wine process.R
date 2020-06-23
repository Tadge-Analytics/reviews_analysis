
# attempt based off David Robinson's Wine Reviews analysis.
# However, that analysis was for a wine "score" not a yes/no response, such as this is.

library(tidytext)

review_rating_words <- prepared_files_data %>%
  unnest_tokens(word, review) %>%
  anti_join(stop_words, by = "word") %>%
  filter(str_detect(word, "[a-z]"))


review_rating_words %>%
  count(word, sort = TRUE) %>%
  head(20) %>%
  mutate(word = fct_reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  coord_flip()

library(widyr)

review_words_filtered <- review_rating_words %>%
  distinct(review_id, word) %>%
  add_count(word) %>%
  filter(n >= 100)

review_words_filtered %>%
  pairwise_cor(word, review_id, sort = TRUE)

library(Matrix)

review_word_matrix <- review_words_filtered %>%
  cast_sparse(review_id, word)

review_ids <- as.integer(rownames(review_word_matrix))

scores <- prepared_files_data$result[review_ids]

library(glmnet)

doParallel::registerDoParallel()

cv_glmnet_model <- cv.glmnet(review_word_matrix, scores, parallel = TRUE)


plot(cv_glmnet_model)


lexicon <- cv_glmnet_model$glmnet.fit %>%
  tidy() %>%
  filter(lambda == cv_glmnet_model$lambda.1se) %>% 
  filter(term != "(Intercept)") %>%
  select(word = term, coefficient = estimate)


lexicon %>%
  arrange(coefficient) %>%
  group_by(direction = ifelse(coefficient < 0, "Negative", "Positive")) %>%
  top_n(16, abs(coefficient)) %>%
  ungroup() %>%
  mutate(word = fct_reorder(word, coefficient)) %>%
  ggplot(aes(word, coefficient, fill = direction)) +
  geom_col() +
  coord_flip() +
  labs(x = "",
       y = "Estimated effect of the word on the score",
       title = "What words are predictive of a review rating?")


