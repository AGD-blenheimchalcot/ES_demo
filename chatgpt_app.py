import streamlit as st
import openai, os
from dotenv import load_dotenv

load_dotenv()
from spacy_pos import *
import pandas as pd
import re

openai.api_key = os.getenv("API_KEY")


def chatgpt_response(corpus_questions, users_text):
    template = f"""You are an English Language Tutor.Now evaluate the given text "{users_text}". Give me the numerical count of times the below given conditions are true for the above text by a user :- {corpus_questions}.
      format must be statement:count """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=template,
        temperature=0.01,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].text


st.title(":green[ES] - AI tutor")

corpus_questions = """
Can make simple statements with 'it's/it is'.
Can use regular nouns in the plural form.
Can use subject personal pronouns.
Can ask basic questions using 'What's your ...?'
Can use the correct form of 'be' with singular and plural nouns.
Can make basic statements with subject + verb + object.
Can use subject pronouns with the correct form of the verb 'be' in the simple present.
Can say their own age using 'I'm [number]'.
Can use the verb 'be' in the simple present with adjectives.
Can use 'and' to link nouns and noun phrases.
Can use common forms of 'have' in the present tense.
Can form questions with 'what' and 'who' and answer them.
Can use 'this is' for an introduction.
Can use 'a/an' with the names of jobs.
Can ask where other people are using 'Where is/are …?'
Can use possessive adjectives such as 'my', 'your', etc.
Can ask someone's age using 'How old …?'
Can use common irregular nouns in the plural form.
Can use 'a/an' with single countable nouns.
Can make affirmative statements using the present simple without time reference.
Can ask yes/no questions using the present simple.
Can use the present simple to refer to daily routines.
Can ask wh- questions using the present simple.
Can use basic prepositions of place with nouns and noun phrases.
Can use 'a/an' with jobs to talk about work and professions.
Can make negative statements using the present simple.
Can use verbs in the imperative.
Can use a range of time expressions with whole numbers (+ 'o'clock').
Can use personal pronouns as objects and complements.
Can use 'there' + 'be' to express presence/absence.
Can ask about the price of something using 'How much is/are …?'
Can use the present simple to refer to likes, dislikes and opinions.
Can ask a range of wh- questions.
Can say where they and other people are using a few basic prepositions.
Can give dates (e.g. their date of birth) using ordinal numbers in the form day-month-year or month-day-year.
Can use 's to express possession with singular nouns.
Can use 'but' to link clauses and sentences.
Can construct short answers to questions in the present simple using the verb 'do'.
Can use common forms of 'have got' (BrE) in the present tense.
Can tell when to use the present simple and when to use the present continuous.
Can use 'I'd like …/I want …' to express wants and wishes.
Can use 'can' to refer to ability in the present.
Can use negative forms of the simple past.
Can ask yes/no questions using the past tense of verbs.
Can use the present continuous to refer to events at the time of speaking.
Can use s' to express possession with plural nouns.
Can use 'that' and 'this' as determiners relating to people or objects.
Can make affirmative statements using common regular past simple forms.
Can use 'and' with verbs and verb phrases.
Can use 'at' as a preposition of time.
Can ask about quantities using 'how much/many' with count and uncountable nouns.
Can use a range of prepositions of place.
Can use an adjective as a subject complement after a linking verb.
Can make affirmative statements using common irregular past simple forms.
Can use a range of common time expressions with 'past'/'to' and fractions.
Can use 'want to' + infinitive to express intentions.
Can ask wh- questions using the past tense of verbs.
Can ask for information about time, measurement, size etc. with 'how' + adjective/quantifier.
Can use 'like/hate/love' with the '-ing' forms of verbs.
Can place adjectives in the correct position (before nouns).
Can use 'was' and 'were' with a range of complement phrases.
Can use the definite article to refer to a specific person, thing, or situation.
Can use uncountable (mass) nouns with no quantifier or an appropriate quantifier.
Can use 'this' with time expressions referring to the present or future.
Can link clauses and sentences with a range of basic connectors.
Can use 'after' as a preposition in time expressions.
Can form questions with 'How often' in the present tense.
Can give, deny or ask about permission in the present and near future with 'can'.
Can make basic polite requests with 'could'.
Can describe times exactly using numbers from 1 to 59 (+ 'past/to').
Can use a range of common adverbs of frequency.
Can ask questions about how to do things.
Can use 'can't' to decline offers and invitations.
Can use 'it' as a dummy (impersonal) subject when talking about weather conditions.
Can make requests and offers with 'would like' + nouns and noun phrases.
Can use the correct preposition ('on' or 'at') with various common time expressions.
Can use 'can' and 'can't' with verbs of perception.
Can use 'Let's …' for suggestions and invitations.
Can make offers, requests, and suggestions using 'can'.
Can use '(not) here' and '(not) there' to refer to presence and absence.
Can form questions with 'whose'.
Can form questions with 'what' and 'which' as adjectives.
Can use 'will' to ask questions about the future.
Can use 'because' with verb phrases to refer to causes and reasons.
Can use 'please' in the correct position with imperative verb forms.
Can use a range of common prepositions of movement.
Can use a range of common time markers for the past, present and future.
Can use a range of common adverbs of movement and direction."""

def extract_counts_from_response(response):
    counts, statements = [], []
    for line in response.split("\n"):
        if line:
            count = int(line.split(": ")[-1])
            statement = line.split(": ")[0]

            counts.append(count)
            statements.append(statement)
    return counts, statements


user_text = st.text_input("Your input text")
button = st.button("Submit")
if button:
    vocab_df = vocab_score(user_text)

    data_container = st.container()

    with data_container:
        table, plot = st.columns(2)
        with table:
            st.write("Each word with its POS tag (Using Spacy Library) :-")
            st.dataframe(vocab_df)
        with plot:
            st.write("GSE scores for vocab list :-")
            gse_vocab_df = get_gse_vocab_score(user_text)
            st.dataframe(gse_vocab_df)
    with st.spinner("please wait while we are assessing your input"):
        response = chatgpt_response(corpus_questions, user_text)
        numbers_ls, statements_ls = extract_counts_from_response(response)
        statements_df = pd.DataFrame()
        statements_df["statements"] = statements_ls
        statements_df["counts"] = numbers_ls
        st.dataframe(statements_df)
        st.bar_chart(statements_df, x="statements", y="counts")

