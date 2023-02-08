from fastapi import APIRouter
from spam_classification import SpamClassifier
import yaml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

with open("./configuration.yaml", "r") as stream:
    try:
        configuration = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Please provide the YAML FILE ")

router = APIRouter(prefix="/classifier", tags=["classifier"], responses={404: {"description": "Not found"}})


@router.post("/generate_model")
async def run_model():
    spam_cls = SpamClassifier(configuration["dataset"])
    spam_cls.make_corpus()
    word_vectors, y_labels = spam_cls.vectorize()

    x_train, x_test, y_train, y_test = train_test_split(word_vectors, y_labels, test_size=0.2, random_state=10)

    spam_detect_model = MultinomialNB().fit(x_train, y_train)

    pickle.dump(spam_detect_model, open(configuration["model_filename"], "wb"))

    return {"status": 200, "description": "model trained and saved successfully"}


@router.get("/spam_classifier/{sentence}")
async def evaluate_sentence(sentence: str):
    spam_detector_model = pickle.load(open((configuration["model_filename"])))
    is_spam = spam_detector_model.predict(sentence)
    is_spam_dict = {0: "Not spam", 1: "Spam Email"}
    return {"status": 200, "description": is_spam_dict[is_spam]}
