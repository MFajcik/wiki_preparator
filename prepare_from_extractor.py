import configparser
import os
import re
import string
import sys
from collections import Counter

import time
import nltk
from nlpfit.preprocessing.nlp_io import read_word_chunks
from nlpfit.preprocessing.tools import DotDict
from nltk.corpus import stopwords
from nltk.corpus.reader import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import spacy

from nlpfit.preprocessing.preprocessor import preprocess_file


# Deprecated, this is too slow
def wiki_text_processor_nltk(chunk: str, tools, opts, logger, wordcounter) -> (str, dict):
    processed_chunk = ""
    chunk = rgx_remove_doctags_and_title.sub("", chunk)
    tokenized_chunk = tools.sentence_tokenizer(chunk)
    for sentence in tokenized_chunk:
        words = tools.word_tokenizer(sentence)
        is_tag_needed = opts.postag_words or opts.lemmatize_words
        w_list = tools.tagger(words) if is_tag_needed else words
        for t in w_list:
            w = t[0] if is_tag_needed else t
            # Filter stopwords and punctuation
            if opts.remove_stop_words and w in stopWords:
                continue

            if opts.remove_puncuation and w in punctuation:
                continue

            if opts.postag_words and opts.lemmatize_words:
                wnet_POS = get_wordnet_pos(t[1])
                output = "{}_{}".format(w.tools.lemmatizer.lemmatize(t[0], wnet_POS), t[1])
            elif opts.postag_words:
                output = "{}_{}".format(t[0], t[1])
            elif opts.lemmatize_words:
                wnet_POS = get_wordnet_pos(t[1])
                output = tools.lemmatizer.lemmatize(t[0], wnet_POS)
            else:
                output = t

            if opts.to_lowercase:
                output = output.lower()
            if opts.replace_nums and output.replace('.', '', 1).isdigit():
                output = config["Options"]["NUM_replacement"]
            processed_chunk += "%s " % (output)
            if opts.count_words:
                wordcounter[output] = wordcounter.get(output, 0) + 1
        processed_chunk += "\n"
    return processed_chunk, wordcounter


def wiki_text_processor_spacy(chunk: str, tools, opts, logger, wordcounter) -> (str, dict):
    processed_chunk = ""
    chunk = rgx_remove_doctags_and_title.sub("", chunk)
    doc = nlp(chunk, disable=['ner', 'parser'])
    for sentence in doc.sents:
        for w in sentence:
            # Some phrases are automatically tokenized by Spacy
            # i.e. New York, in that case we want New_York in our dictionary
            word = "_".join(w.text.split())

            if word.isspace() or word == "":
                continue
            if opts.remove_stop_words and word.lower() in stopWords:
                continue

            if opts.remove_puncuation and word in punctuation:
                continue
            # Spacy lemmatized I,He/She/It into artificial
            # -PRON- lemma, which is unwanted
            if opts.postag_words and opts.lemmatize_words:
                raw_lemma = w.lemma_ if w.lemma_ != '-PRON-' else w.lower_
                word_lemma = "_".join(raw_lemma.split())
                output = "{}_{}".format(word_lemma, w.pos_)
            elif opts.postag_words:
                output = "{}_{}".format(word, w.pos_)
            elif opts.lemmatize_words:
                raw_lemma = w.lemma_ if w.lemma_ != '-PRON-' else w.lower_
                word_lemma = "_".join(raw_lemma.split())
                output = word_lemma
            else:
                output = word

            if opts.to_lowercase and not opts.lemmatize_words:
                output = output.lower()
            if opts.replace_nums and output.replace('.', '', 1).isdigit():
                output = config["Options"]["NUM_replacement"]
            processed_chunk += "%s " % (output)
            if opts.count_words:
                wordcounter[output] = wordcounter.get(output, 0) + 1
        processed_chunk += "\n"
    return processed_chunk, wordcounter


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


# 10485760
def en_worker(idx, s_offset, e_offset, tmpdir, opts, logger, chunk_size=1000000,
              text_processor=wiki_text_processor_nltk, num_of_processes=None):
    ofilename = "{}_".format(idx) + os.path.basename(opts.ofile) if opts.ofile else None
    wordcounter = Counter() if opts.count_words else None

    preprocessing_tools = DotDict()
    preprocessing_tools.tagger = nltk.pos_tag
    preprocessing_tools.lemmatizer = WordNetLemmatizer()
    preprocessing_tools.sentence_tokenizer = nltk.sent_tokenize
    preprocessing_tools.word_tokenizer = nltk.word_tokenize
    if opts.ofile:
        if idx == 0:
            # Benchmark perf
            starttime = time.time()
            processed_so_far = 0
            with open(os.path.join(tmpdir, ofilename), mode="w") as of:
                for chunk in read_word_chunks(opts.ifile, chunk_size, s_offset, e_offset):
                    preprocessed, wordcounter = text_processor(chunk, preprocessing_tools, opts, logger, wordcounter)
                    of.write(preprocessed)
                    if num_of_processes is not None:
                        processed_so_far += chunk_size
                        delta = time.time() - starttime
                        print("\r Processing speed ~{} kB/s".format(
                            processed_so_far / delta / 1e3 * num_of_processes))
        else:
            with open(os.path.join(tmpdir, ofilename), mode="w") as of:
                for chunk in read_word_chunks(opts.ifile, chunk_size, s_offset, e_offset):
                    preprocessed, wordcounter = text_processor(chunk, preprocessing_tools, opts, logger, wordcounter)
                    of.write(preprocessed)
    else:
        for chunk in read_word_chunks(opts.ifile, chunk_size, s_offset, e_offset):
            preprocessed, wordcounter = text_processor(chunk, preprocessing_tools, opts, logger, wordcounter)
    return wordcounter


def str_to_bool(i):
    return i == "True" or i == "true" or i == "T" or i == "t" or i == "1"


# Script expects an input corpus recieved from wiki extractor
# Corpus still containst basic tags out of context (sentence) words
# i.e. at the start of each paragraph
# </doc>
# <doc id="39" revid="30540869" url="https://en.wikipedia.org/wiki?curid=39" title="Albedo">
# Albedo
#
# Albedo () (, meaning "whiteness") is the measure of the diffuse reflection of solar radiation

if __name__ == "__main__":
    nlp = spacy.load('en')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    corpus_path = config["Data"]["in_path"]
    output = config["Data"]["out_path"]
    stopWords = set(stopwords.words('english'))
    punctuation = list(string.punctuation) + ["``", "''"]
    #
    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('wordnet')
    # Remove
    # <doc tags>
    # article title
    # sentences in parantheses
    r_remove_doctags_and_title = r"(<doc.*>(\n)*(\w+)(\n)*)|(</doc>)"
    rgx_remove_doctags_and_title = re.compile(r_remove_doctags_and_title)

    print("Lemmatize: {}".format(str_to_bool(config["Options"]["lemmatize"])))
    print("remove_stop_words: {}".format(str_to_bool(config["Options"]["remove_stop_words"])))
    print("remove_punct: {}".format(str_to_bool(config["Options"]["remove_punct"])))
    print("postag_words: {}".format(str_to_bool(config["Options"]["postag_words"])))
    time_taken, wordcounter = preprocess_file(corpus_path, output,
                                              count_words=True,
                                              text_processor=wiki_text_processor_spacy,  # wiki_text_processor_nltk,
                                              process_worker=en_worker,
                                              num_of_processes=int(config["Options"]["processes"]),
                                              lemmatize_words=str_to_bool(config["Options"]["lemmatize"]),
                                              remove_stop_words=str_to_bool(config["Options"]["remove_stop_words"]),
                                              remove_puncuation=str_to_bool(config["Options"]["remove_punct"]),
                                              postag_words=str_to_bool(config["Options"]["postag_words"]),
                                              tmpdir=config["Options"]["tm_folder_name"])

    import operator

    wordcounter_l = sorted(wordcounter.items(), key=operator.itemgetter(1), reverse=True)

    with open(config["Data"]["vocab"], 'w') as outf:
        for key, value in wordcounter_l:
            outf.write("%s %d\n" % (key, value))

    with open(config["Data"]["statistics_f"], 'w') as outf:
        outf.write("Total number of words: {}".format(sum(wordcounter.values())))
