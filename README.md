## Text Classification

This project uses text from www.Reddit.com to develop classification models that can predict a topic in a binary choice. A pair of "subreddits" was chosen: AskScience (questions about science) and EatCheapAndHealthy (questions and suggestions about making and eating inexpensive and healthy food). The vocabulary of these two was, not surprisingly, generally different. In the second phase, the AskScience subreddit was compared to a more similar one: Ask Historians (moderated questions about history).

### Files Available
[Presentation](https://docs.google.com/presentation/d/1-KijFcVglbwLC9koAC-dR9h6iyXQCPLeQt7bTAqaxGY/edit#slide=id.g561541372a_0_55)


Jupyter Notebooks:

[Data Acquisition](./Project%203%20-%20Reddit%20-%20Data%20Acquisition.ipynb)

[AskScience vs. EatCheapAndHealthy](./Project%203%20-%20Reddit%20-%20EatCheap%20vs.%20AskScience.ipynb)

[AskScience vs. AskHistorians](./Project%203%20-%20Ask%20Historians%20vs%20Ask%20Science.ipynb)


### Data Acquisition
I used the method of extracting subreddit pages by using the url www.reddit.com/r/SUBREDDIT.json. I created a function that saves the needed variables to a dataframe using arguments for the subreddit name and the number of pages to load. I also separately used the pushshift.io API to easily save 1,000 posts for a subreddit to a dataframe. This method saves all of the columns associated with each post. However, I ended up using the fields subreddit, title, and selftext, so I did not need this method. 
I spent some time considering which subreddits to compare and looked throught the list of subreddits. Many subreddits have posts that include just a title and a link. I decided to use subreddits that have none (or very few) of these, based on the rules of the subreddit. I initially picked a subreddit about food and cooking (EatCheapAndHealthy) and one about science (AskScience), thinking that it would be fairly easy to distinguish the two. After I collected up to 1,000 posts for each subreddit I saved the separate dataframes using the to_pickle method so that they could be easily retrieved later.

### Data Cleaning
I made sure there were no duplicate posts by using the dropdupicates() property. The 'selftext' field is plain text with HTML removed. However, I removed line break characters ("\n"). I combined the title and text fields into one text field for analysis in order to have more words per observation, and also because in many cases the title is just the first part of the text. I combined the data from the two subreddits into a single dataframe. I created a dummy variable, science, with a value of 1 if the text came from the askscience subreddit and 0 otherwise (that is, from the EatCheapAndHealthy subreddit).

I used "stopwords" from the NLTK package. (However, many of the best-performing models did not use stopwords.) I used the FreqDistVisualizer() function from the [Yellowbrick Machine Learning Visualization](https://www.scikit-yb.org/en/latest/) package to graph the top n (default of 50) most frequent words for each subreddit. I noticed that there were some that should not be included, such as  parts of URLs (http,com,org) and a few common words not included in stopwords (e.g., "could").

To create a more attractive and interesting display of the most common words in each subreddit (after excluding the stopwords), I created wordclouds from the wordcloud package. This is the wordcloud for AskScience posts:


![WordCloud of AskScience words](https://git.generalassemb.ly/PaulSchimek/submissions/blob/master/project3/images/science.png)


The most common words in EatCheapAndHealthy were completely different than AskScience:

![WordCloud of EatCheapAndHealthy words](https://git.generalassemb.ly/PaulSchimek/submissions/blob/master/project3/images/eatcheap.png)

### Modeling
After the data cleaning, there were more posts in AskScience, so the baseline accuracy was 67%. I used test-train-split and divdied the data into a 70% training set and a 30% test set.

I used the CountVectorizer() module from scikit-learn to make word frequencies into separate variables for modeling. I initially used a logistic regression model. After tweaking the parameters (C, use of stopwords, etc.), the best model had an accuracy of 96.8%. Using the TF-IDF (term frequency, inverse document frequency) vectorizer increased the accuracy to 98.2%.

I also tried using Naive Bayes models. The NB model using the Bernouilli distribution and the basic count vectorizer had, after tweaking the parameters, an accuracy of 97.3%. This was improved by using the Multinomial distribution and the TF-IDF vectorizer, which provided an accuracy of 99.1%. 

I also used a support vector machines (SVM) model. This model is less computationally efficient than the other two, but is reasonably fast if one limits the number of features to less than 1,000 (this restriction does not reduce the SVM model's accuracy). The SVM model accuracy is particulary dependent on selecting hyper-parameters. The best SVM model had an accuracy of 96.1% -- good, but worse than any of the other models.

### Misclassified Data
With the best model (Naive Bayes with TF-IDF), there were only 4 misclassified posts in the testing set (30% of the data):

  - two very short posts inluding one that was only nine words.
  - an advertising link.
  - an AskScience post about chicken and egg allergies that contained the word "eating."
  

### Ask Science vs. Ask Historians

It was fairly easy to produce a very accurate model of two subreddits that were very different. What about subreddits with more similar topics? I decided to compare AskScience to AskHistorians. The latter is also a moderated subreddit that does not allow link-only posts and has specific requirements about the types of questions allowed. I repeated the data acquistion process for AskHistorians, and combined it with the existing AskScience data. I repeated the data cleaning. An examination of the wordcloud for AskHistorians shows that it has at least two words, "know" and "question", that overlap with the most common words from AskScience:

![WordCloud of AskHistorians words](https://git.generalassemb.ly/PaulSchimek/submissions/blob/master/project3/images/history.png)

The baseline accuracy was 67% (there were fewer than 1,000 posts available). The best model was again Naive Bayes with a multinomial distribution and the TF-IDF vectorizer. This produced an accuracy of 96.6%. There were only 17 misclassified posts in the testing data. 

### Conclusions
With 1,000 examples of text, we can create a classifier with 97% or better accuracy. This was true both for topics that seem very different (food / cooking and science), and also for more similar, but still distinct, topics (science vs. history). The EDA showed that the categories had distinct vocabularies. 

For both sets of comparisons, the Naive Bayes model using a multinomial distribution and a TF-IDF vectorizer performed best. Tweaking the parameters increases accuracy a few percentage points. However, this adjustment may be specific to the sample at hand and may not be robust to new samples.

An examination of the misclassified posts suggests that some posts were hard to classify because they were very short or were not on topic (e.g., promotions or announcements). There are also some posts that included vocabulary from another topic area and were thus very difficiult for machine classification.

It would make sense to limit the observations to posts of a certain minimum length (e.g., more than 40 words) in order to have a better chance at classifying them correctly.
