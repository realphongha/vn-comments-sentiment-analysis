import argparse
import sys
sys.path.append("..")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, render_template, request, redirect
from crawler import crawl_page, create_browser
from selenium.common.exceptions import InvalidArgumentException, TimeoutException
from datasets.preprocess import preprocess_sentence
from predict import PhoBertPredictor


app = Flask(__name__, template_folder="templates",
            static_folder="static")


class Comment:
    COLOR = {"POS": "green", "NEG": "red", "NEU": "grey"}

    def __init__(self, text, result, confidence):
        self.text = text
        self.result = result
        self.confidence = confidence
        self.color = Comment.COLOR[result]


def phoBERT(raw_data):
    data = preprocess_sentence(raw_data)
    return phoBertPredictor.predict(data)


# def NB(raw_data):
#     if opt.preprocess:
#         data = preprocess(raw_data)
#     else:
#         data = raw_data
#     data = texts_segmented(data)
#     predict_labels, confidences = nb.inference(data)
#     return predict_labels, confidences


# def SVM(raw_data):
#     if opt.preprocess:
#         data = preprocess(raw_data)
#     else:
#         data = raw_data
#     data = texts_segmented(data)
#     predict_labels, confidences = l_svm.inference(data)
#     return predict_labels, confidences


@app.route("/sentence", methods=["POST", "GET"])
def sentence():
    if request.method == "POST":
        s = request.form["sentence"]
        model = request.form["model"]
        data = [s]
        if model == "phoBERT":
            predict_label, confidences = phoBERT(data)
        # elif model == "SVM":
        #     predict_label, confidences = SVM(data)
        # elif model == "NB":
        #     predict_label, confidences = NB(data)
        else:
            raise NotImplementedError("Invalid model!")
        result = Comment(s, opt.cls[int(predict_label[0])], 
                         round(confidences[0] * 100, 3))
        return render_template("sentence.html", result=result)
    else:
        return render_template("sentence.html")


@app.route("/vnexpress", methods=["POST", "GET"])
def vnexpress():
    if request.method == "POST":
        browser = create_browser()
        results = []
        url = request.form["url"]
        model = request.form["model"]
        try:
            original_comments, title = crawl_page(url, browser)
            if model == "phoBERT":
                predict_labels, confidences = phoBERT(original_comments)
            elif model == "SVM":
                predict_labels, confidences = SVM(original_comments)
            elif model == "NB":
                predict_labels, confidences = NB(original_comments)
            else:
                raise NotImplementedError("Invalid model")
            for i in range(len(original_comments)):
                results.append(
                    Comment(original_comments[i], 
                            opt.cls[int(predict_labels[i])], 
                            round(confidences[i] * 100, 3)))
        except InvalidArgumentException:
            return render_template("vnexpress.html", message="Please enter a valid URL!")
        except TimeoutException:
            return render_template("vnexpress.html", message="Cannot crawl comments from %s!" % url)
        return render_template("vnexpress.html", results=results, num_results=len(results), title=title)
    else:
        return render_template("vnexpress.html")


@app.route("/")
def index():
    return redirect("/vnexpress")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', type=str, nargs="+",
                        default=["POS", "NEG", "NEU"], 
                        help='classes')
    parser.add_argument('--device', default='cpu', 
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, 
                        default='../log/VNEX2021--2021-12-06--08-34/ckpt_best.pth', 
                        help='path to checkpoint')
    opt = parser.parse_args()
    
    phoBertPredictor = PhoBertPredictor(opt.weights, opt.device, len(opt.cls))

    app.run(debug=False)
