""" Computational Models for Named Entity Recognition, WS20/21
    ISCL Program, University of Tübingen
    Milan Miletić - Final Project
    (for detailed description see the attached paper)
"""

import spacy
from spacy.lang.sr import Serbian
from spacy.pipeline import EntityRuler
from spacy.util import minibatch, compounding
from spacy import displacy
from pathlib import Path
import re
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from spacy.gold import GoldParse
from spacy.scorer import Scorer


def ann_to_spacy(input_dir):
    """ Converts the data from .ann files to the format required by spaCy

    Positions of tokens in .ann files tell us about their position in respect to the WHOLE TEXT,
    and not about their position within the sentence;
    The idea is to first save all entities from the .ann files, then get rid of XML tags from .txt files,
    then segment the text into sentences, then recognize previously saved tokens within the sentences,
    and finally add each sentence to the data structure used for training.

    We want to end up with this format:
    patterns = [{"label": "ORG", "pattern": "Apple"},
                {"label": "GPE", "pattern": "San Francisco"}]

    Parameters
    -------------
    input_dir: path to the directory containing .ann and .txt files

    Returns
    -------------
    2-tuple where the first element represents the patterns and the second a set of entity labels
    """

    # loading .ann files (located in input_dir) and saving annotated tokens with annotations
    patterns = []
    labels = set()

    'standoff: T79	ROLE 9311 9320	председник'
    ann_regex = re.compile(r'T\d+\t(\w+) (\d+) (\d+)\t([\w ]+)', re.UNICODE)

    for filename in os.listdir(input_dir):
        if filename.endswith('.ann'):
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')
                for line in lines:
                    m = ann_regex.search(u'%s' % line)
                    if m:
                        label = m.group(1)  # ROLE
                        pattern = m.group(4)  # председник
                        patterns.append({"label": label, "pattern": pattern})
                        labels.add(label)

    return patterns, labels


def split_train_test(input_dir, patterns, test_size=0.2):
    """ Splits the converted data into train and test set and updates the model

    After extracting the patterns in the spaCy-required format from .ann files,
    we can split the data into the train and test set and get ready for the training process

    Parametres
    -------------
    input_dir: path to the directory containing .ann and .txt files
    patterns: patterns as returned by ann_to_spacy()
    test_size: percentage of the data reserved for evaluation (20% by default)

    Returns (3 tuple)
    -------------
    train: training data
    test: testing data
    nlp: updated model
    """

    # Blank model for Serbian (no info about entities)
    nlp = Serbian()
    ruler = EntityRuler(nlp)  # EntityRuler for saving entities
    ruler.add_patterns(patterns)  # Adding the entities
    nlp.add_pipe(ruler)  # Adding the ruler to the model

    # Now the blank model recognizes the entities that were found in the files

    # Training data is in the following format:
    # TRAIN_DATA = [("Uber blew through $1 million a week", {'entities': [(0, 4, 'ORG')]})]

    # Adding a 'sentencizer' in order for the model to know how to segment the text into sentences
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    data = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                # immediately remove XML tags
                content = re.sub('<[^<]+>', '', f.read())

                # segmentation into sentences
                doc = nlp(content)
                for sent in doc.sents:
                    # now a sentence is a single document, so that we can index the entities within it
                    doc = nlp(sent.text)
                    entities = []
                    # adding only those sentences that contain entities
                    if len(doc.ents) > 0:
                        for ent in doc.ents:
                            entities.append((ent.start_char, ent.end_char, ent.label_))

                        data.append((sent.text, {'entities': entities}))

    data = np.array(data)
    # Data format: [['sentence', {'entities': [(start, end, label)]}]]
    train, test = train_test_split(data, test_size=test_size, random_state=10)

    return train, test, nlp


def train(data, nlp, labels, n_iter=30, drop=0.5, batch_from=4.0, batch_to=32.0, batch_compound=1.001):
    """ Trains the model using spaCy's standard procedure

    The default values of the function's parametres represent the values
    of the hyperparametres in the most successful model

    Parametres
    -------------
    data: training data as returned by split_train_test()
    nlp: the model as returned by split_train_test()
    labels: entity labels as returned by ann_to_spacy()
    n_iter: number of iterations (default: 30)
    drop: dropout rate (default: 0.5)
    batch_from: initial batch size (default: 4.0)
    batch_to: final batch size (default: 32.0)
    batch_compound: rate of batch size acceleration (1.001)

    Returns
    -------------
    nlp - the trained model
    """
    # Standard routine for training a model
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Adding annotations for training
    for label in labels:
        ner.add_label(label)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(data)
            losses = {}
            batches = minibatch(data, size=compounding(batch_from, batch_to, batch_compound))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, drop=drop, losses=losses)
            print("Losses", losses)

    return nlp


def save_to_disk(nlp, model_dir):
    """ Saves the model to disk

    Parametres
    -------------
    nlp - the model that you want to save
    model_dir - path to the directory you want to save the model to (created for you if non-existing)
    """
    # Saving the model
    if not Path(model_dir).exists():
        Path(model_dir).mkdir()
    nlp.to_disk(model_dir)
    print("Saved model to", model_dir)


def eval_standard(ner_model, examples, label=None):
    """ Evaluates the model using the standard evaluation metrics

    If label parameter is not defined, the scores will be calculated for the whole document,
    otherwise it will calculate the scores for the selected label only

    Parametres
    -------------
    ner_model - the model that you want to evaluate
    examples - data reserved for evaluation (test set)
    label - entity label (one of: PERS, ROLE, ORG, LOC, WORK, EVENT, DEMO)

    Returns (a 3-tuple)
    -------------
    precision, recall and F1-score
    """
    scorer = Scorer()
    for input_, ents in examples:
        doc_gold_text = ner_model.make_doc(input_)
        ann = ents['entities']
        gold = GoldParse(doc_gold_text, entities=ann)
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    if label:
        label_scores = scorer.scores['ents_per_type'][label.upper()]
        return label_scores['p'], label_scores['r'], label_scores['f']
    else:
        return scorer.scores['ents_p'], scorer.scores['ents_r'], scorer.scores['ents_f']


def eval_slen(ner_model, examples):
    """ Evaluates the model based on sentence length

    Parametres
    -------------
    ner_model - the model that you want to evaluate
    examples - data reserved for evaluation (test set)

    Returns (a 3-tuple)
    -------------
    where each element is another 3-tuple that consists of the
    precision, recall and F1-score for each bucket
    """
    sLen_buckets = [[], [], []]  # buckets for sentence length ([<5], [5-10], [>10])
    nlp = Serbian()

    # fill the buckets
    for entry in examples:
        input_ = entry[0]  # sentence
        ents = entry[1]  # entities

        sLen = len([token.text for token in nlp(input_)])
        if sLen < 5:
            sLen_buckets[0].append(entry)
        elif sLen <= 10:
            sLen_buckets[1].append(entry)
        else:
            sLen_buckets[2].append(entry)

    # eval for sLen <5
    s1 = Scorer()
    for input_, ents in sLen_buckets[0]:
        doc_gold_text = ner_model.make_doc(input_)
        ann = ents['entities']
        gold = GoldParse(doc_gold_text, entities=ann)
        pred_value = ner_model(input_)
        s1.score(pred_value, gold)
    r1 = s1.scores['ents_p'], s1.scores['ents_r'], s1.scores['ents_f']

    # eval for sLen 5-10
    s2 = Scorer()
    for input_, ents in sLen_buckets[1]:
        doc_gold_text = ner_model.make_doc(input_)
        ann = ents['entities']
        gold = GoldParse(doc_gold_text, entities=ann)
        pred_value = ner_model(input_)
        s2.score(pred_value, gold)
    r2 = s2.scores['ents_p'], s2.scores['ents_r'], s2.scores['ents_f']

    # eval for sLen >10
    s3 = Scorer()
    for input_, ents in sLen_buckets[2]:
        doc_gold_text = ner_model.make_doc(input_)
        ann = ents['entities']
        gold = GoldParse(doc_gold_text, entities=ann)
        pred_value = ner_model(input_)
        s3.score(pred_value, gold)
    r3 = s3.scores['ents_p'], s3.scores['ents_r'], s3.scores['ents_f']

    return r1, r2, r3


def eval_elen(ner_model, examples):
    """ Evaluates the model based on entity length

        Parametres
        -------------
        ner_model - the model that you want to evaluate
        examples - data reserved for evaluation (test set)

        Returns (a 3-tuple)
        -------------
        where each element is another 3-tuple that consists of the
        precision, recall and F1-score for each bucket
        """
    eLen_buckets = [[], [], []]  # buckets for entity length (in words) ([1], [2], [>2])
    nlp = Serbian()

    # fill the buckets
    for input_, ents in examples:
        ents_0, ents_1, ents_2 = [], [], []  # entity lists for corresponding buckets
        for ent in ents['entities']:
            eLen = len([token.text for token in nlp(input_[ent[0]:ent[1]])])
            if eLen == 1:
                ents_0.append(ent)
            elif eLen == 2:
                ents_1.append(ent)
            else:
                ents_2.append(ent)

        if ents_0:
            eLen_buckets[0].append([input_, {'entities': ents_0}])
        if ents_1:
            eLen_buckets[1].append([input_, {'entities': ents_1}])
        if ents_2:
            eLen_buckets[2].append([input_, {'entities': ents_2}])

    # eval for eLen = 1
    s1 = Scorer()
    for input_, ents in eLen_buckets[0]:
        doc_gold_text = ner_model.make_doc(input_)
        ann = ents['entities']
        gold = GoldParse(doc_gold_text, entities=ann)
        pred_value = ner_model(input_)
        s1.score(pred_value, gold)
    r1 = s1.scores['ents_p'], s1.scores['ents_r'], s1.scores['ents_f']

    # eval for eLen = 2
    s2 = Scorer()
    for input_, ents in eLen_buckets[1]:
        doc_gold_text = ner_model.make_doc(input_)
        ann = ents['entities']
        gold = GoldParse(doc_gold_text, entities=ann)
        pred_value = ner_model(input_)
        s2.score(pred_value, gold)
    r2 = s2.scores['ents_p'], s2.scores['ents_r'], s2.scores['ents_f']

    # eval for eLen >2
    s3 = Scorer()
    for input_, ents in eLen_buckets[2]:
        doc_gold_text = ner_model.make_doc(input_)
        ann = ents['entities']
        gold = GoldParse(doc_gold_text, entities=ann)
        pred_value = ner_model(input_)
        s3.score(pred_value, gold)
    r3 = s3.scores['ents_p'], s3.scores['ents_r'], s3.scores['ents_f']

    return r1, r2, r3


if __name__ == "__main__":

    # patterns, labels = ann_to_spacy('input')
    # tr, test, nlp = split_train_test('input', patterns)
    # model = train(tr, nlp, labels, n_iter=30, drop=0.5, batch_from=4.0, batch_to=32.0, batch_compound=1.001)
    # save_to_disk(nlp, 'model')

    # model - default values
    ner_model = spacy.load('./model')
    patterns, labels = ann_to_spacy('input')
    tr, te, nlp = split_train_test('input', patterns)

    text = ' '.join([sent for sent, _ in te])
    doc = nlp(text)
    displacy.render(doc, style='ent')
    displacy.serve(doc, style='ent')
