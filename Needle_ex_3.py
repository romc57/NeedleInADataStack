import numpy as np
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re

BOOK_PATH = 'allice.txt'
ADJ = 'ADJ'
NOUN = 'NOUN'
PROPN = 'PROPN'
PRON = 'PRON'


def text_from_file(file_path):
    book = str()
    with open(file_path, encoding='utf8') as input_data:
        for line in input_data:
            if line == 'CHAPTER I.\n':
                break
        for line in input_data:
            if line == 'THE END \n':
                break
            book += line
    return book


def count_tokens(doc, stop_words_out):
    token_count = dict()
    for token in doc:
        if stop_words_out and (token.is_stop or token.is_space or token.is_punct):
            continue
        else:
            if token.text not in token_count:
                token_count[token.text] = 1
            else:
                token_count[token.text] += 1
    return token_count


def count_stemmed_token(doc, punct_out):
    token_count = dict()
    for token in doc:
        if punct_out and (token.is_space or token.is_punct):
            continue
        if token.lemma_ not in token_count:
            token_count[token.lemma_] = 1
        else:
            token_count[token.lemma_] += 1
    return token_count


def find_pos(doc):
    pos_dictionary = dict()
    for token in doc:
        if token.text not in pos_dictionary and len(token.text) > 2:
            pos_dictionary[token.text] = [token.pos_]
        elif token.text in pos_dictionary and token.pos_ not in pos_dictionary[token.text]:
            pos_dictionary[token.text].append(token.pos_)
    return pos_dictionary


def extract_adj_nouns_phrases(doc):
    adj_nouns_lst = create_by_order_lst(doc, [ADJ, NOUN, PROPN, PRON])
    adj_nouns_phrases = list()
    i = 0
    word, pos = adj_nouns_lst[i]
    while i < len(adj_nouns_lst):
        curr_lst = list()
        while pos == ADJ:
            curr_lst.append(word)
            i += 1
            if i < len(adj_nouns_lst):
                word, pos = adj_nouns_lst[i]
            else:
                break
        while pos == NOUN or pos == PROPN or pos == PRON:
            curr_lst.append(word)
            i += 1
            if i < len(adj_nouns_lst):
                word, pos = adj_nouns_lst[i]
            else:
                break
        curr_phrase = " ".join(curr_lst)
        adj_nouns_phrases.append(curr_phrase)
    return adj_nouns_phrases


def create_by_order_lst(doc, pos_lst):
    lst_pos = list()
    for token in doc:
        if token.pos_ in pos_lst and not token.is_stop:
            lst_pos.append((token.text, token.pos_))
    return lst_pos


def adj_nouns_in_order(doc):
    lst_pos = list()
    i = 0
    while i < (len(doc)):
        token = doc[i]
        while token.pos_ == ADJ:
            lst_pos.append((token.text, i))
            i += 1
            if i < len(doc):
                token = doc[i]
            else:
                break
            while token.pos_ == PROPN or token.pos_ == NOUN or token.pos_ == PRON:
                lst_pos.append((token.text, i))
                i += 1
                if i < len(doc):
                    token = doc[i]
                else:
                    break
        i += 1
    return create_sequences(lst_pos)


def create_sequences(lst_pos):
    output = list()
    i = 0
    while i < len(lst_pos) - 1:
        word, index = lst_pos[i]
        word_2, index_2 = lst_pos[i + 1]
        curr_seq = [word]
        while index_2 - index == 1:
            curr_seq.append(word_2)
            i += 1
            if i < len(lst_pos) - 1:
                word, index = lst_pos[i]
                word_2, index_2 = lst_pos[i + 1]
            else:
                break
        i += 1
        if len(curr_seq) > 1:
            output.append(curr_seq)
    return output


def get_top_twenty(dictionary):
    sorted_dic = {k: v for k, v in sorted(dictionary.items(), key=lambda item: -item[1])}
    top_twenty = list()
    keys = sorted_dic.keys()
    for i, key in enumerate(keys):
        if i > 19:
            break
        top_twenty.append(key)
    return top_twenty, sorted_dic


def plot_counts(sorted_dictionary, graph_name):
    values = sorted_dictionary.values()
    x_axis = np.arange(len(sorted_dictionary)) + 1
    fig, ax = plt.subplots()
    ax.bar(x_axis, values)
    plt.title(graph_name)
    ax.set_ylabel('Log Frequency')
    ax.set_xlabel('Log Rank')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()


def phrases_count(phrases):
    token_count = dict()
    for token in phrases:
        token = tuple(token)
        if token not in token_count:
            token_count[token] = 1
        else:
            token_count[token] += 1
    return token_count


def get_pos_homographs(dictionary):
    sorted_dic = {k: v for k, v in sorted(dictionary.items(), key=lambda item: -len(item[1]))}
    items = list(sorted_dic.items())
    reversed_items = list(reversed(items))
    top_10 = list()
    down_10 = list()
    for i in range(len(items)):
        if i > 9:
            break
        top_10.append(items[i])
        down_10.append(reversed_items[i])
    return top_10, down_10


def get_nouns_lst(doc):
    return np.array(create_by_order_lst(doc, [PROPN]))[:, 0].tolist()


def word_cloud_plot(word_lst):
    comment_words = " ".join(word_lst) + " "
    stopwords = set(STOPWORDS)
    word_cloud = WordCloud(width=800, height=800,
                           background_color='white',
                           stopwords=stopwords,
                           min_font_size=10).generate(comment_words)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def find_duplicates_word(text):
    duplicate_words = list()
    founds = re.findall(r'\b(\w+)( *\,* +\.* *)\1\b', text)
    for found, sep in founds:
        duplicate_words.append(found + sep + found)
    return duplicate_words


def run_all(file_path, nlp_model):
    book_txt = text_from_file(file_path)
    analyzed_book = nlp_model(book_txt)
    # task b)
    count_token_dictionary = count_tokens(analyzed_book.doc, False)
    top_twenty, sorted_count_token = get_top_twenty(count_token_dictionary)
    plot_counts(sorted_count_token, '(b) Full Tokens')
    print(top_twenty)
    # # task c)
    count_token_dictionary_no_stops = count_tokens(analyzed_book.doc, True)
    top_twenty_no_stop, sorted_count_token_no_stop = get_top_twenty(count_token_dictionary_no_stops)
    plot_counts(sorted_count_token_no_stop, '(c) No Stops')
    print(top_twenty_no_stop)
    # # task d)
    count_token_dictionary_lemma = count_stemmed_token(analyzed_book.doc, True)
    top_twenty_stemmed, sorted_count_lemma = get_top_twenty(count_token_dictionary_lemma)
    plot_counts(sorted_count_lemma, '(d) Stemmed')
    print(top_twenty_stemmed)
    # task e)
    adj_noun_phrases = adj_nouns_in_order(analyzed_book.doc)
    count_phrase = phrases_count(adj_noun_phrases)
    top_twenty_phrases, sorted_phrases_dic = get_top_twenty(count_phrase)
    plot_counts(sorted_phrases_dic, '(e) adj + noun phrases')
    print(top_twenty_phrases)
    # task g)
    pos_dictionary = find_pos(analyzed_book.doc)
    top_10, down_10 = get_pos_homographs(pos_dictionary)
    print(top_10)
    print(down_10)
    # task h
    nouns = get_nouns_lst(analyzed_book.doc)
    word_cloud_plot(nouns)
    # task i)
    duplicates = find_duplicates_word(book_txt)
    print(duplicates)


if __name__ == '__main__':
    model = spacy.load('en_core_web_sm')
    run_all(BOOK_PATH, model)
