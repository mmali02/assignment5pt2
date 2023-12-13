from src.classifier.classifier_models import AbstractClassifier, FeatureSet
import nltk
from nltk.tokenize import word_tokenize
from operator import itemgetter
from typing import Any, Iterable, List, Dict

nltk.download('punkt')
nltk.download('stopwords')


class Feature:
    """Feature used for the classification of an object.

    Attributes:
        _name (str): Human-readable name of the feature (e.g., "word_exists")
        _value (any): Machine-readable value of the feature (e.g., True)
    """

    def __init__(self, name, value=True):
        self._name: str = name
        self._value: any = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> any:
        return self._value

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, Feature):
            return False
        else:
            return self._name == other.name and self._value == other.value

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self._name} = {self._value}"

    def __hash__(self) -> int:
        return hash((self._name, self._value))


class FeatureSet:
    """A set of features that represent a single object. Optionally includes the known class of the object.

    Attributes:
        _feat (set[Feature]): A set of features that define this object for the purposes of a classifier
        _clas (str | None): Optional attribute set as the pre-defined classification of this object
    """

    def __init__(self, features: set[Feature], known_clas=None):
        self._feat: set[Feature] = set(features)
        self._clas: str | None = known_clas

    @property
    def feat(self):
        return self._feat

    @property
    def clas(self):
        return self._clas

    @classmethod
    def build(cls, source_object: str, known_clas=None) -> 'FeatureSet':
        """
    Build and return an instance of FeatureSet given a source object (movie review text).

    Tokenize the text and create features based on word occurrences and pre-classified positive/negative words.

    :param source_object: Movie review text.
    :param known_clas: Pre-defined classification of the source object.
    :param negative_words: List of additional negative words.
    :return: An instance of FeatureSet built based on the source object.
    """
        # Tokenize the input text to extract individual words
        words = word_tokenize(source_object)

        # Original features based on word occurrences
        original_features = {Feature(word): True for word in words}

        # Additional features for pre-classified positive and negative words
        positive_words = {'good', 'excellent', 'amazing', 'great', 'funny', 'laugh', 'thrilling', 'fun', 'terrific', 'special', 'best', 'incredible'}

        negative_words = {'bad', 'poor', 'horrible', 'meh', 'terrible', 'not', 'why', 'how', 'off', 'criticism', 'fuck',
                          '?', 'shit', 'what', 'mediocre', 'worse'}
        # Add features based on the presence of positive and negative words in the text
        for word_feature in positive_words:
            original_features[Feature(f"positive_{word_feature}")] = word_feature in words

        for word_feature in negative_words:
            original_features[Feature(f"negative_{word_feature}")] = word_feature in words
        # Return an instance of FeatureSet with the built features and known classification
        return cls(original_features, known_clas)


class NaiveBayesTextClassifier(AbstractClassifier):
    """Abstract definition for an object classifier."""

    def __init__(self):
        super().__init__()
        self.class_priors = {}
        self.feature_likelihoods = {}

    def gamma(self, a_feature_set: FeatureSet) -> str:
        """
            Calculate the posterior probability for each class based on the given feature set.

            :param a_feature_set: An instance of FeatureSet containing relevant features for classification.
            :return: The predicted class based on the highest posterior probability.
            """
        # Initialize a dictionary to store posterior probabilities for each class
        posterior_probs = {}
        # Calculate posterior probabilities for each class
        for clas, prior in self.class_priors.items():
            # Initialize likelihood with a default value of 0.5
            likelihood = 0.5
            # Iterate through each feature in the feature set
            for feature in a_feature_set.feat:
                # Check if the feature has likelihood information for the current class
                if feature.name in self.feature_likelihoods and clas in self.feature_likelihoods[feature.name]:
                    # Update likelihood based on the feature's likelihood for the current class
                    likelihood *= self.feature_likelihoods[feature.name][clas]
            # Calculate the overall posterior probability for the current class
            posterior_probs[clas] = prior * likelihood

        # Make a classification decision based on the class with the highest posterior probability
        predicted_class = max(posterior_probs, key=posterior_probs.get)
        # Return the predicted class
        return predicted_class

    def present_features(self, top_n: int = 1) -> List[Dict[str, Any]]:
        """
            Present the top N informative features and their probabilities based on the trained classifier.

            :param top_n: The number of top informative features to display. Default is 1.
            :return: A list of dictionaries containing feature information, class, and probability.
            """
        # Check if the classifier has been trained
        if not self.feature_likelihoods:
            print("Classifier has not been trained yet. Call the train method first.")
            return []

        # List to store informative features and their probabilities
        informative_features = []

        # Iterate over each feature and class
        for feature_name, class_probs in self.feature_likelihoods.items():
            for clas, probability in class_probs.items():
                # Append feature information to the list
                informative_features.append({
                    'feature': Feature(feature_name),
                    'class': clas,
                    'probability': probability
                })

        # Sort the informative features by probability in descending order
        informative_features.sort(key=itemgetter('probability'), reverse=True)

        # Return the top N informative features
        return informative_features[:top_n]

    def train(self, training_set: Iterable[FeatureSet]) -> 'NaiveBayesTextClassifier':
        """
           Train the Naive Bayes text classifier using the provided training set.

           :param training_set: An iterable containing instances of FeatureSet for training the classifier.
           :return: The trained NaiveBayesTextClassifier instance.
           """
        # Calculate class priors
        total_samples = 0
        class_counts = {}
        # Count occurrences of each class in the training set
        for data_point in training_set:
            total_samples += 1
            class_counts[data_point.clas] = class_counts.get(data_point.clas, 0) + 1
        # Calculate and store class priors
        for clas, count in class_counts.items():
            self.class_priors[clas] = count / total_samples

        # Calculate feature likelihoods
        feature_counts = {}
        class_totals = {}
        # Iterate through the training set to count features for each class
        for data_point in training_set:
            class_totals[data_point.clas] = class_totals.get(data_point.clas, 0) + 1
            for feature in data_point.feat:
                feature_name = feature.name

                # Handle pre-classified positive and negative word features separately
                if feature_name.startswith("positive_") or feature_name.startswith("negative_"):
                    if feature_name not in feature_counts:
                        feature_counts[feature_name] = {}
                    if data_point.clas not in feature_counts[feature_name]:
                        feature_counts[feature_name][data_point.clas] = 0
                    feature_counts[feature_name][data_point.clas] += feature.value

                # Handle other features
                else:
                    if feature_name not in feature_counts:
                        feature_counts[feature_name] = {}
                    if data_point.clas not in feature_counts[feature_name]:
                        feature_counts[feature_name][data_point.clas] = 0
                    feature_counts[feature_name][data_point.clas] += 1
        # Calculate and store feature likelihoods using Laplace smoothing
        for feature_name, class_counts in feature_counts.items():
            if feature_name not in self.feature_likelihoods:
                self.feature_likelihoods[feature_name] = {}
            for clas, count in class_counts.items():
                # Use Laplace smoothing
                self.feature_likelihoods[feature_name][clas] = (count + 1) / (class_totals[clas] + len(feature_counts))
        # Return the trained classifier instance
        return self


