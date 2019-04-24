import os
import re
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import CuDNNLSTM, LSTM, Dense, Embedding, Bidirectional
from keras.utils import plot_model
from keras import Sequential
import matplotlib.pyplot as plt


train_nrows = None
max_length = 66
dictionary_size = 120000

dirpath = os.path.realpath('./input')
filepath_data_train = os.path.join(dirpath, 'train.csv')
filepath_data_test = os.path.join(dirpath, 'test.csv')
filepath_embeddings_glove = os.path.join(dirpath, 'embeddings/glove.840B.300d/glove.840B.300d.txt')
filepath_embeddings_paragram = os.path.join(dirpath, 'embeddings/paragram_300_sl999/paragram_300_sl999.txt')
filepath_embeddings_wiki = os.path.join(dirpath, 'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
filepath_missing = 'missing_data.csv'
filepath_missing_prep = 'missing_data_prep.csv'


def get_time():
     return '[' + datetime.datetime.now().strftime("%H:%M:%S") + ']'


def multireplace(text, replacements):
     """
     Given a string and a replacement map, it returns the replaced string.
     :param str string: string to execute replacements on
     :param dict replacements: replacement dictionary {value to find: 
value to replace}
     :rtype: str
     """
     # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
     # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
     # 'hey ABC' and not 'hey ABc'
     substrs = sorted(replacements, key=len, reverse=True)
	 
     for p in substrs:
         text = text.replace(p, replacements[p])
		 
     '''
     # Create a big OR regex that matches any of the substrings to replace
     pattern = '|'.join(map(re.escape, substrs))
     if pattern not in regexps:
         regexp = re.compile(pattern)
         regexps[pattern] = regexp
     regexp = regexps[pattern]

     # For each match, look up the new string in the replacements
     return regexp.sub(lambda match: replacements[match.group(0)], text)
	 '''
     return text


def preprocess(text):
     specials = ["’", "‘", "´", "`"]
     for s in specials:
         text = text.replace(s, "'")

     contraction_mapping = {"what's":"what is", "What's":"What is", "i'm":"i am", "I'm":"i am", "isn't":"is not", "Isn't":"is not", "i've":"i have", "I've":"i have", "you've":"you have", "aren't":"are not", "Aren't":"are not", "won't":"will not", "Won't":"will not", "they're":"they are", "They're":"they are", "he's":"he is", "He's":"he is", "haven't":"have not", "shouldn't":"should not", "Shouldn't":"should not", "wouldn't":"would not", "Wouldn't":"would not", "who's":"who is", "Who's":"who is", "there's":"there is", "There's":"there is", "wasn't":"was not", "Wasn't":"was not", "she's":"she is", "hasn't":"has not", "Hasn't":"has not", "couldn't":"could not", "we're":"we are", "We're":"we are", "i'll":"i will", "I'll":"i will", "i'd":"i would", "I'd":"i would", "how's":"how is", "How's":"how is", "let's":"let us", "Let's":"let us", "weren't":"were not", "Weren't":"were not", "they've":"they have", "we've":"we have", "We've":"we have", "hadn't":"had not", "Hadn't":"had not", "you'd":"you would", "where's":"where is", "Where's":"where is", "'the":"the", "'The":"the", "'i":"i", "'I":"i", "would've":"would have", "“the":"the", "“The":"the", "“i":"i", "“I":"i","they'll":"they will", "They'll":"they will", "he'll":"he will", "He'll":"he will"}
     text = multireplace(text, contraction_mapping)

     posessive_mappings = {"Trump's":"trump", "trump's":"trump", "Obama's":"obama", "obama's":"obama", "Google's":"google", "google's":"google", "India's":"india", "india's":"india", "Russia's":"russia", "russia's":"russia", "Israel's":"israel", "israel's":"israel", "Korea's":"korea", "korea's":"korea", "China's":"china", "china's":"china", "America's":"america", "america's":"america", "canada's":"canada", "Canada's":"canada", "pakistan's":"pakistan", "Pakistan's":"pakistan", "iran's":"iran", "Iran's":"iran", "japan's":"japan", "Japan's":"japan", "UK's":"uk", "uk's":"uk", "britain's":"britain", "Britain's":"britain", "usa's":"usa", "USA's":"usa", "germany's":"germany", "Germany's":"germany", "someone's":"someone", "else's":"else", "today's":"today", "people's":"people", "women's":"women", "men's":"men", "world's":"world", "earth's":"earth", "Earth's":"earth", "country's":"country", "person's":"person", "quora's":"quora", "Quora's":"quora", "man's":"man", "woman's":"woman", "God's":"God", "company's":"company", "father's":"father", "mother's":"mother", "child's":"child", "girl's":"girl", "boy's":"boy", "wife's":"wife", "husband's":"husband", "year's":"year", "dog's":"dog", "friend's":"friend", "children's":"children", "driver's":"driver", "government's":"government", "everyone's":"everyone", "girlfriend's":"girlfriend", "boyfriend's":"boyfriend", "other's":"other", "modi's":"modi", "Modi's":"modi", "son's":"son", "daughter's":"daughter", "sister's":"sister", "cat's":"cat", "asperger's":"asperger", "Asperger's":"asperger", "alzheimer's":"alzheimer", "Alzheimer's":"alzheimer", "jehovah's":"jehovah", "Jehovah's":"jehovah", "einstein's":"einstein", "Einstein's":"einstein", "clinton's":"clinton", "Clinton's":"clinton", "king's":"king", "life's":"life", "parents'":"parents", "hitler's":"hitler", "Hitler's":"hitler", "newton's":"newton", "Newton's":"newton", "amazon's":"amazon", "Amazon's":"amazon", "xavier's":"xavier", "Xavier's":"xavier", "king's":"king", "King's":"king", "university's":"university", "University's":"university", "student's":"student", "Putin's":"putin", "putin's":"putin", "mom's":"mom", "baby's":"baby", "guy's":"guy", "president's":"president", "President's":"president"}
     text = multireplace(text, posessive_mappings)

     meaning_mapping = {"quorans":"quora", "Quorans":"quora", "90's":"90s", "80's":"80s", "70's":"70s", "Brexit":"britain exit", "brexit":"britain exit", "master's":"masters", "Master's":"masters", "mcdonald's":"mcdonalds", "McDonald's":"mcdonalds", "one's":"someone", "cryptocurrencies":"bitcoin", "bachelor's":"bachelors", "Bachelor's":"bachelors", "demonetisation":"demonetization", "qoura":"quora", "Qoura":"quora", "Qur'an":"quran", "qur'an":"quran", "sjws":"sjw", "SJWs":"sjw"}
     text = multireplace(text, meaning_mapping)
     return text


def init_tokenizer(filepath, prep=False):
     global dictionary_size, train_nrows
     print(get_time(), "Training data load started...")
     if train_nrows is not None:
         df = pd.read_csv(filepath, nrows=train_nrows)
     else:
         df = pd.read_csv(filepath)
     print(get_time(), "Training data loaded:", df.shape[0])

     if prep:
         tqdm.pandas()
         df["question_text"] = df["question_text"].progress_apply(lambda x: preprocess(x))

     print(get_time(), "Tokenizer start...")
     texts = df['question_text'].values
     tokenizer = Tokenizer(num_words=dictionary_size)
     tokenizer.fit_on_texts(texts)
     print(get_time(), "Tokenizer fit. Total words:", len(tokenizer.word_index))

     return tokenizer


def get_missing_words(word_index, word_count, embedding_dict):
     total = 0.
     missing = 0.
     missing_words = {}
     for word in word_index:
         total += word_count[word]
         if word not in embedding_dict:
             missing_words[word] = word_count[word]
             missing += word_count[word]
     missing_content_ratio = (100. * missing)/total
     missing_words_ratio = (100. * len(missing_words))/len(word_index)
     print(get_time(), 'Missing content: {:.2f}%'.format(missing_content_ratio) + '. ' +
                       'Missing words: {:.2f}%'.format(missing_words_ratio))
     print('-----')
     for s in sorted(missing_words.items(), reverse=True, key=lambda item: item[1])[:50]:
         print(s[0], ': ', s[1])
     print('-----')
     return missing_words, missing_content_ratio, missing_words_ratio

def load_embeddings_glove(filepath, word_index):
     print(get_time(), "Start load embeddings (glove)...")
     embedding_dict = {}
     missing_words = {}
     for line in open(filepath, encoding='utf8'):
         split = line.split(' ')
         word = split[0]
         if len(split) == 301 and word in word_index:
             embedding_dict[word] = np.array(split[1:], dtype=np.float32)
     print(get_time(), "Finished load embeddings (glove):", len(embedding_dict))
     return embedding_dict


def load_embeddings_paragram(filepath, word_index):
     print(get_time(), "Start load embeddings (paragram)...")
     embedding_dict = {}
     for line in open(filepath, encoding="utf8", errors='ignore'):
         split = line.split(' ')
         word = split[0]
         if len(split) == 301 and word in word_index:
             embedding_dict[word] = np.array(split[1:], dtype=np.float32)
     print(get_time(), "Finished load embeddings (paragram):", len(embedding_dict))
     return embedding_dict


def load_embeddings_wiki(filepath, word_index):
    print(get_time(), "Start load embeddings (wikinews)...")
    embedding_dict = {}
    for line in open(filepath, encoding="utf8", errors='ignore'):
        split = line.split(' ')
        word = split[0]
        if len(split) == 301 and word in word_index:
            embedding_dict[word] = np.array(split[1:], dtype=np.float32)
    print(get_time(), "Finished load embeddings (wikinews):", len(embedding_dict))
    return embedding_dict


def write_missing_result(filepath_output, prep=False):
    global filepath_embeddings_glove
    global filepath_embeddings_paragram 
    global filepath_embeddings_wiki
    global filepath_data_train

    tokenizer = init_tokenizer(filepath_data_train, prep)
    # glove
    glove = load_embeddings_glove(filepath_embeddings_glove, tokenizer.word_index)
    g_words, g_miss1, g_miss2 = get_missing_words(tokenizer.word_index, tokenizer.word_counts, glove)
    # paragram
    paragram = load_embeddings_paragram(filepath_embeddings_paragram, tokenizer.word_index)
    p_words, p_miss1, p_miss2 = get_missing_words(tokenizer.word_index, tokenizer.word_counts, paragram)
    # wiki
    wiki = load_embeddings_wiki(filepath_embeddings_wiki, tokenizer.word_index)
    w_words, w_miss1, w_miss2 = get_missing_words(tokenizer.word_index, tokenizer.word_counts, wiki)

    missing_data = np.array([['Glove','Paragram','Wiki'],
                             [g_miss1,p_miss1,w_miss1],
                             [g_miss2,p_miss2,w_miss2]])
    df_missing = pd.DataFrame(data=missing_data[1:,:], columns=missing_data[0,:])
    df_missing.to_csv(filepath_output, index=False)


def bar_plot(title, data, filepath_png):
    fig, ax = plt.subplots()
    index = np.arange(2)
    bar_width = 0.20
    opacity = 0.5

    missing_content_g = ax.bar(index, data[0], bar_width, alpha=opacity, color='b', label='Glove')
    missing_content_p = ax.bar(index + bar_width, data[1], bar_width, alpha=opacity, color='r', label='Paragram')
    missing_content_p = ax.bar(index + 2*bar_width, data[2], bar_width, alpha=opacity, color='g', label='Wikinews')

    ax.set_xlabel('Before and after text preprocessing')
    ax.set_ylabel('Missing %')
    ax.set_title(title)
    #ax.set_xticks(index + bar_width / 2)
    #ax.set_xticklabels(('A', 'B'))
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    ax.legend()

    fig.tight_layout()
    #plt.show()
    plt.savefig(filepath_png)


def write_missing_results():
    global filepath_missing
    global filepath_missing_prep
    write_missing_result(filepath_missing, prep=False)
    write_missing_result(filepath_missing_prep, prep=True)


def read_missing_results_and_plot():
    global filepath_missing
    global filepath_missing_prep

    df_missing_n = pd.read_csv(filepath_missing)
    g_n_1, g_n_2 = df_missing_n.iloc[0,0], df_missing_n.iloc[1,0]
    p_n_1, p_n_2 = df_missing_n.iloc[0,1], df_missing_n.iloc[1,1]
    w_n_1, w_n_2 = df_missing_n.iloc[0,2], df_missing_n.iloc[1,2]

    df_missing_p = pd.read_csv(filepath_missing_prep)
    g_p_1, g_p_2 = df_missing_p.iloc[0,0], df_missing_p.iloc[1,0]
    p_p_1, p_p_2 = df_missing_p.iloc[0,1], df_missing_p.iloc[1,1]
    w_p_1, w_p_2 = df_missing_p.iloc[0,2], df_missing_p.iloc[1,2]

    bar_plot('Missing content in embeddings', [[g_n_1, g_p_1],[p_n_1, p_p_1],[w_n_1, w_p_1]], 'missing_content.png')
    bar_plot('Missing words in embeddings', [[g_n_2, g_p_2],[p_n_2, p_p_2],[w_n_2, w_p_2]], 'missing_words.png')


def model_plot():
    vector_size, max_length, dictionary_size = 300, 66, 120000
    embedding_matrix = np.zeros([dictionary_size, vector_size])
    model = Sequential()
    model.add(Embedding(dictionary_size, vector_size, 
                        weights=[embedding_matrix], 
                        trainable=False, 
                        input_length=max_length))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    plot_model(model, to_file='model.png')



#write_missing_results()
#read_missing_results_and_plot()
model_plot()





