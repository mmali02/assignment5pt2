# Assignment 5: Open-ended
**Westmont College Fall 2023**

**CS 128 Information Retrieval and Big Data**

*Assistant Professor* Mike Ryu (mryu@westmont.edu) 

## Link to presentation:
https://docs.google.com/presentation/d/1SuW4E10XwUxWgl30xAzSBgmsFA-aaStfGcs3kAg19sw/edit?usp=sharing

## Author Information
* **Names**: Morgan Mali
* **Emails**: mmali@westmont.edu

## License information: 
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

## Problem Description:
I adapted code from a previous assignment, using a Naive Bayes classifier in order to classify movie reviews into positive and negative categories. In a similar fashion, this code is using a Naive Bayes classifier to distinguish between positive and negative tweets.


## Guide of Code:

# Feature class
Feature used for the classification of an object.

    Attributes:
        _name (str): Human-readable name of the feature (e.g., "word_exists")
        _value (any): Machine-readable value of the feature (e.g., True)
    """

# Feature Set class: 
A set of features that represent a single object. Optionally includes the known class of the object.

    Attributes:
        _feat (set[Feature]): A set of features that define this object for the purposes of a classifier
        _clas (str | None): Optional attribute set as the pre-defined classification of this object

  # Build function:
        Build and return an instance of FeatureSet given a source object (e.g., tweet from twitter_samples).

        Tokenize the text and create features based on word occurrences and pre-classified positive/negative words.

        :param source_object: Text from the source object (e.g., tweet).
        :param known_clas: Pre-defined classification of the source object.
        :return: An instance of FeatureSet built based on the source object.
        
# Naive Bayes Classifier

#   Gamma function: 
            Calculate the posterior probability for each class based on the given feature set.

            :param a_feature_set: An instance of FeatureSet containing relevant features for classification.
            :return: The predicted class based on the highest posterior probability.

#   Present features function:
            Present the top N informative features and their probabilities based on the trained classifier.

            :param top_n: The number of top informative features to display. Default is 1.
            :return: A list of dictionaries containing feature information, class, and probability.

            
#   Train function: 
           Train the Naive Bayes text classifier using the provided training set.

           :param training_set: An iterable containing instances of FeatureSet for training the classifier.
           :return: The trained NaiveBayesTextClassifier instance.

# Runner: 

# Load tweet data: 

                Load tweet data and split it into training and testing sets.

                Args:
                    test_size: Proportion of the dataset to include in the test split.

                Returns:
                    Tuple of training and testing data.
                


## Citing sources: 
“based on the code above, how would you change this code:” 
prompt. ChatGPT, 13 Dec. version 3.5, OpenAI, 14 Dec. 2023, chat.openai.com/chat.

“based off of this code above, how would you change this runner 
below, now that we are using twitter_samples:” prompt. ChatGPT, 13 Dec. version 3.5, OpenAI, 14 Dec. 2023, chat.openai.com/chat.

“how would you change this code based off of twitter samples?” 
prompt. ChatGPT, 13 Dec. version 3.5, OpenAI, 14 Dec. 2023, chat.openai.com/chat.

“how would you change this code specifically?” prompt. ChatGPT, 
13 Dec. version 3.5, OpenAI, 14 Dec. 2023, chat.openai.com/chat.

“if i want to change this code for twitter_samples, in order to 
distinguish between all categories of twitter_samples, what would you do?” prompt. ChatGPT, 13 Dec. version 3.5, OpenAI, 14 Dec. 2023, chat.openai.com/chat.

“Twitter Howto¶.” Twitter, 
www.nltk.org/howto/twitter.html#:~:text=NLTK’s%20Twitter%20corpus%20currently%20contains,fileids(). Accessed 12 Dec. 2023. 

