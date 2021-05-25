#!/usr/bin/python

import argparse

from Evaluator import Evaluator
from LanguageModel import LanguageModel
from TextProcessor import TextProcessor


def main():
    parser = argparse.ArgumentParser(description='Script to train a language model')
    parser.add_argument("--train", default="../data/train.csv", type=str,
                        help="text file containing the training data")
    parser.add_argument("--test_input", default="../data/test.csv", type=str,
                        help="text file containing the test data")

    args = parser.parse_args()

    """ Init components """
    """ Use your own langauge model and text processor"""
    processor = TextProcessor()
    lm = LanguageModel(1)
    eval = Evaluator(lm, processor, args.voc)

    """ Load training data and train language model"""
    text = getText(args.train)

    """ Process text by the Text Processor"""
    prepro = processor.process(text)

    """ Process text by the Text Processor"""
    print("Train language model ....")
    lm.train(prepro)
    print("Language model trained")

    print("Test data")
    test = getText(args.test_input)
    reference = getText(args.test_output)
    prepro_reference = processor.process(reference)
    lm.getPPL(prepro_reference)
    eval.eval(test, reference)


def getText(filename):
    f = open(filename, encoding="utf-8")

    text = [l.strip().split() for l in f.readlines()]
    text = [item for sublist in text for item in sublist]
    f.close()
    return text


if __name__ == "__main__":
    main()
