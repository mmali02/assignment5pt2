from typing import List

from sklearn.model_selection import train_test_split

from src.classifier.perro_classifier.perro_classifier_models import Feature, FeatureSet, NaiveBayesTextClassifier
from nltk.corpus import movie_reviews
import nltk
import os
import random


nltk.download('movie_reviews')


class TwitterRunner:
    def __init__(self):
        self.classifier = NaiveBayesTextClassifier()
        self.data = self.load_movie_reviews_data()
        self.output_directory = "out"  # Output directory

    def load_movie_reviews_data(self, test_size=0.2) -> tuple[List[FeatureSet], List[FeatureSet]]:
        """
                Load movie reviews data and split it into training and testing sets.

                Args:
                    test_size: Proportion of the dataset to include in the test split.

                Returns:
                    Tuple of training and testing data.
                """

        positive_fileids = movie_reviews.fileids('pos')
        negative_fileids = movie_reviews.fileids('neg')
        all_fileids = positive_fileids + negative_fileids
        min_samples = min(len(positive_fileids), len(negative_fileids))
        random.shuffle(all_fileids)

        reviews = [
            FeatureSet.build(movie_reviews.raw(fileid), known_clas='pos' if fileid in positive_fileids else 'neg') for
            fileid in all_fileids[:min_samples]]

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(reviews, test_size=test_size, random_state=42)

        return train_data, test_data

    def train_classifier(self):
        self.classifier.train(self.data[0])  # Training data

    def classify_reviews(self) -> List[tuple]:
        results = []
        correct_predictions = 0

        for review in self.data[1]:  # Testing data
            predicted_label = self.classifier.gamma(review)
            results.append((review.clas, predicted_label))
            if predicted_label == review.clas:
                correct_predictions += 1

        # Calculate accuracy
        accuracy = correct_predictions / len(self.data[1])  # Testing data
        return results, accuracy

    def determine_top_features(self, top_n: int = 30) -> List[str]:
        return [feature['feature'].name for feature in self.classifier.present_features(top_n)]

    def print_results_to_file(self, results: List[tuple], accuracy: float, top_features: List[str],
                              output_file: str = "perro_runner_results.txt"):
        output_path = os.path.join(self.output_directory, output_file)

        # Calculate counts for true positives, true negatives, false positives, and false negatives
        true_positives = sum((actual == 'pos' and predicted == 'pos') for actual, predicted in results)
        true_negatives = sum((actual == 'neg' and predicted == 'neg') for actual, predicted in results)
        false_positives = sum((actual == 'neg' and predicted == 'pos') for actual, predicted in results)
        false_negatives = sum((actual == 'pos' and predicted == 'neg') for actual, predicted in results)

        with open(output_path, 'w') as file:
            file.write("PerroRunner Results:\n")
            file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
            file.write("\nTop Features:\n")
            for feature in top_features:
                file.write(f"{feature}\n")
            file.write("\nCounts:\n")
            file.write(f"True Positives: {true_positives}\n")
            file.write(f"True Negatives: {true_negatives}\n")
            file.write(f"False Positives: {false_positives}\n")
            file.write(f"False Negatives: {false_negatives}\n")

    def run(self):
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self.train_classifier()
        results, accuracy = self.classify_reviews()

        # Extract length feature for printing
        length_feature = next(iter(self.data[1][0].feat))  # Assuming the first feature is the length
        top_features = self.determine_top_features()
        top_features.append(f"Review Length: {length_feature.value}")

        # Extract pre-classified positive and negative word features for printing
        positive_word_features = [feature for feature in top_features if feature.startswith("positive_")]
        negative_word_features = [feature for feature in top_features if feature.startswith("negative_")]

        print(f"\nPre-classified Positive Words: {positive_word_features}")
        print(f"Pre-classified Negative Words: {negative_word_features}")

        self.print_results_to_file(results, accuracy, top_features)
        self.print_console_results(results, accuracy, top_features)

    @staticmethod
    def print_console_results(results: List[tuple], accuracy: float, top_features: List[str]):
        print(f"TwitterRunner Accuracy: {accuracy * 100:.2f}%")
        print("\nTop Features:")
        for feature in top_features:
            print(feature)
        print("\nReview Details:")
        for actual, predicted in results:
            print(f"Actual: {actual}, Predicted: {predicted}")


if __name__ == "__main__":
    tweet_runner = TwitterRunner()
    tweet_runner.run()
