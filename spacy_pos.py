import spacy
import streamlit as st

nlp = spacy.load("en_core_web_lg")
import pandas as pd

global df
df = pd.read_csv("gsevocab1.csv")


def get_gse_vocab_score(input_text):
    words = input_text.split()
    matches = df[df["vocab"].isin(words)]
    return matches

def vocab_score(input_text):
    text = input_text.lower()
    output_sentence = "".join(filter(lambda x: x not in ",:@%*!.{}()~-", text))
    doc = nlp(output_sentence)
    vocab_df = pd.DataFrame()
    tokens, grammar = [], []
    for token in doc:
        print(token.text, spacy.explain(token.pos_))
        tokens.append(token.text)
        grammar.append(spacy.explain(token.pos_))
    vocab_df["words"] = tokens
    vocab_df["grammar"] = grammar
    return vocab_df
