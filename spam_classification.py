import pandas as pd
import re
import nltk
from tqdm import tqdm

import concurrent.futures
import yaml
import pickle

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

from fastapi import APIRouter

tqdm.pandas()
nltk.download('stopwords')
router = APIRouter(
    prefix="/spam",
    tags=["spam"],
    responses={404: {"description": "Not found"}}
)

with open("./configuration.yaml", "r") as stream:
    try:
        configuration = yaml.safe_load(stream)
        print(configuration)
    except yaml.YAMLError as exc:
        print(exc)
# df = pd.read_csv('./archive/Datasets/final_dataset.csv',
#                  encoding='cp1252')
# print(df.head())
corpus = []
ps = PorterStemmer()


class SpamClassifier:
    def __init__(self, config):
        self.df = pd.read_csv(config["dataset_path"], encoding=config["dataset_encoding"])
        self.corpus = []
        self.max_features = config["max_features"]

    @staticmethod
    def sentence_operations(sentence):
        # print(sentence)
        reviews = re.sub('[^a-zA-Z0-9]', ' ', sentence)
        reviews = reviews.lower()
        reviews = reviews.split()

        reviews = [ps.stem(word) for word in reviews if word not in stopwords.words('english')]
        reviews = ' '.join(reviews)

        return reviews

    def make_corpus(self):
        # to make up the progress faster !! we can use
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(SpamClassifier.sentence_operations, self.df["Email"][i]) for i in range(len(self.df))]

            for review in tqdm(concurrent.futures.as_completed(results)):
                self.corpus.append(review.result())

        return self.corpus

    def vectorize(self):
        cv = CountVectorizer(max_features=5000)  # choosing frequent 5000 words
        word_vectors = cv.fit_transform(self.corpus).toarray()
        label_values = {"FRAUD": 2, "SPAM": 1, "NORMAL": 0}
        df2 = self.df.replace({"Label": label_values})
        y_labels = df2["Label"]

        return word_vectors, y_labels


# checking the y column

# print(df.head())
# label_encoder = LabelEncoder()
# labels = label_encoder.fit_transform((df["Label"]))

# print(labels)

# # 2nd method right now is
# label_values = {"FRAUD": 2, "SPAM": 1, "NORMAL": 0}
# df2 = df.replace({"Label": label_values})
# y = df2["Label"]
# # print(df.head())
# spam_cls = SpamClassifier(configuration["dataset"])
# spam_cls.make_corpus()
# word_vectors, y_labels = spam_cls.vectorize()
#
# x_train, x_test, y_train, y_test = train_test_split(word_vectors, y_labels, test_size=0.2, random_state=10)
#
# spam_detect_model = MultinomialNB().fit(x_train, y_train)
#
# pickle.dump(spam_detect_model, open(configuration["model_filename"], "wb"))
# y_pred = spam_detect_model.predict(x_test)

# # final values
# print(y_pred[:5])
# print(y_test[:5])


@router.get("/is_spam")
async def first_service():
    print("true")
    return {"name": "true to the best of my knowledge"}
