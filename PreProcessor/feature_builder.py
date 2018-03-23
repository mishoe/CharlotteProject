class TextFeature:
    '''
    A class for learning bag of word features from multiple columns of text data passed in from
    a data frame.  Also works for lists.  tranfromation create a word matrix based on new data
    according to the previously fit transformations
    Author: Matthew Davis
    '''
    def __init__(self):
        from nltk.corpus import stopwords
        self.transforms = None
        self.input_names = []
        self.max_features = 200
        self.min_freq = 0.001
        self.ngram_range = (1, 1)
        self.min_len = 3
        self.stop_words = set(stopwords.words('english'))
        self.feature_names = []
        self.generated_column_name = 'x'
        self.par = True

    def fit(self, data):
        '''
        :param data: pandas data frame or list
        :param par: Bool use multi processing pool
        :return: nothing, learns and save the text vectorization method
        '''
        import gc
        data = self.make_df(data)
        self.input_names = list(data.columns)
        gc.collect()
        data_list = [(data[i].tolist(), i) for i in self.input_names]
        if self.par:
            from multiprocessing import Pool
            p = Pool()
            transforms = list(p.map(self.string_list_2_count_vect, data_list))
            p.terminate()
        else:
            transforms = list(map(self.string_list_2_count_vect, data_list))
        self.feature_names = [[t[0] + '.' + w for w in t[1].get_feature_names()] for t in transforms]
        self.feature_names = [item for sublist in self.feature_names for item in sublist]
        self.transforms = dict(transforms)
        gc.collect()

    def string_cleaner(self, doc):
        '''
        :param doc: takes a string a of text,
        preforms min length freq
        :return: a cleaned string of text
        '''
        if doc is None:
            return ''
        else:
            doc = str(doc)
            from keras.preprocessing.text import text_to_word_sequence
            output = text_to_word_sequence(doc)
            output = [word for word in output if len(word) >= self.min_len and word.isdigit() is False]
            output = ' '.join(output)
            return output

    def string_list_2_count_vect(self, x):
        '''
        :param x: a tubple of corpus, string name for the corpus
        :return: returns a count vectorizor
        '''
        corpus = x[0]
        name = x[1]
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(stop_words=self.stop_words, max_features=self.max_features, analyzer='word',
                             min_df=self.min_freq)
        cv.fit(self.string_list_cleaner(corpus))
        return name, cv

    def string_list_cleaner(self, corpus):
        results = map(self.string_cleaner, corpus)
        return list(results)

    def update_stop_words(self, words, append=True):
        '''

        :param words: a list of words
        :param append: logical, to overwrite or append stopwords english from NLTK
        :return:
        '''
        if append:
            self.stop_words.append(words)
        else:
            self.stop_words = words

    def clean_transform_list(self, x):
        '''
        :param x: a tuple of {colname, corpus, countVectorizor)
        :return: a scipy sparse matrix of features and a list of feature names
        '''
        col = x[1]
        corpus = x[0]
        transformer = x[2]
        output_names = [col + '.' + word for word in transformer.get_feature_names()]
        corpus = self.string_list_cleaner(corpus)
        output = transformer.transform(corpus)
        return output, output_names

    def transform(self, data):
        '''
        Transform method
        uses a pre fitted TextFeature Object to transform unseen text data
        :param data: data frame containing text columns in TextFeature.input_names
        :param par: Bool use multi processing pool
        :return: scipy sparse matrix of features, feature_names in TextFeature have the feature names
        '''

        from scipy import sparse
        data = self.make_df(data)
        data_list = [(data[col], col,  self.transforms[col]) for col in self.input_names]
        if self.par:
            from multiprocessing import Pool
            p = Pool()
            output_list = list(p.map(self.clean_transform_list, data_list))
            p.terminate()
        else:
            output_list = list(map(self.clean_transform_list, data_list))
        features = sparse.hstack([f[0] for f in output_list])
        return features

    def make_df(self, data):
        '''
        coerces lists and series to data frames, adds a colname from self.generated_column_name
        :param data: a list, pd.series or pd.data
        :return: a pandas data frame
        '''
        import pandas as pd
        if type(data) is pd.core.series.Series:
            data = pd.DataFrame(data)
        if type(data) is list:
            data = pd.DataFrame(data)
            data.rename(columns={0:self.generated_column_name},  inplace=True)
        if type(data) is pd.core.frame.DataFrame:
            return data
        else:
            print('warning, data type is not a pandas data frame')
            return data


class CatEncoder:
    '''
    a wrapper for category encoder, uses min frequency on fit and saves feature_names
    '''
    def __init__(self):
        self.min_freq = 0.01
        self.input_names = None
        self.feature_names = None
        self.dropped_levels = {}
        self.encoder = None
        self.generated_column_name = 'x'
        self.return_df = False
        self.handle_unknown = 'ignore'
        self.par = True

    def fit(self, data):
        '''
        fits the catagorical encoder, coecces to a pd data frame, save input and feature names
        :param data: a pandas data frame, or list
        :return: nothing, fitted encoder is saved as encoder
        '''
        from category_encoders import OneHotEncoder
        ohe = OneHotEncoder(return_df=self.return_df, handle_unknown=self.handle_unknown)
        x = self.replace_infrequent_df(data)
        self.input_names = x.columns
        ohe.fit(x)
        self.encoder = ohe
        self.feature_names_from_cat_encoder()

    def transform(self, data):
        '''
        applies categorical encoder to un seen data
        :param data: pandas data frame or list
        :return: a scipy sparse matrix or data frame if self.return_df is TRUE
        '''
        x = self.make_df(data)
        x = x[self.input_names]
        if self.return_df:
            return self.encoder.transform(x)
        if self.return_df is False:
            from scipy import sparse
            return sparse.csr_matrix(self.encoder.transform(x))

    def inverse_transform(self, data):
        '''
        reverses the encoder one hot to df, untested
        :param data:
        :return:
        '''
        x = self.make_df(data)
        x = x[self.input_names]
        return self.encoder.inverse_transform(x)

    def replace_infrequent(self, x):
        '''

        :param x: a tuple with (key:, pandas series)
        :return: the orignal key, pandas series with infrequent levels mapped to None
        '''
        key = x[1]
        x = x[0]
        counts = x.value_counts()/x.count()
        infrequent_levels = list(counts.iloc[counts.values < self.min_freq].index)
        x.replace(infrequent_levels, None, inplace=True)
        self.dropped_levels[key] = infrequent_levels
        return key, x

    def replace_infrequent_df(self, data):
        import pandas as pd
        data = self.make_df(data)
        cols = list(data.columns)
        data_list = [(data[i], i) for i in cols]
        if self.par:
            from multiprocessing import Pool
            p = Pool()
            output_dict = dict(list(p.map(self.replace_infrequent, data_list)))
            p.terminate()
        else:
            output_dict = dict(list(map(self.replace_infrequent, data_list)))

        return pd.DataFrame.from_dict(output_dict)

    def feature_names_from_cat_encoder(self):
        output_names = []
        x = self.encoder.category_mapping
        for d in x:
            temp_map = d['mapping']
            temp_map = [y[0] for y in temp_map]
            col = d['col']
            new_names = [col + '.' + t for t in temp_map]
            for n in new_names:
                output_names.append(n)
        self.feature_names = output_names

    def make_df(self, data):
        '''
        coerces lists and series to data frames, adds a colname from self.generated_column_name
        :param data: a list, pd.series or pd.data
        :return: a pandas data frame
        '''
        import pandas as pd
        if type(data) is pd.core.series.Series:
            data = pd.DataFrame(data)
        if type(data) is list:
            data = pd.DataFrame(data)
            data.rename(columns={0: self.generated_column_name}, inplace=True)
        if type(data) is pd.core.frame.DataFrame:
            return data
        else:
            print('warning, data type is not a pandas data frame')
            return data


class DataSetBuilder:
    '''
     a Class build to combine feature extraction for catagories, text and numeric data
     Defaults to using mutliprocessing
     use if __name__ == '__main__': before fit and transform calls on windows
    '''
    def __init__(self, params=None, col_dict=None):
        '''

        :param params:
        :param col_dict: dictionary with keys 'cat_cols',  text_cols', 'imputer_cols'. 'zero_imputer_cols',
        the values are the column names in a pandas data frame to prepreocess
        '''
        from nltk.corpus import stopwords
        self.default_params = {'text_cols': {'max_features': 200, 'min_freq': 0.001, 'ngram_range': (1, 1),
                                             'min_len': 3, 'stop_words': set(stopwords.words('english'))},
                               'cat_cols': {'min_freq': 0.01},
                               'imputer_cols': {'strategy': 'median'}}
        if params is None:
            self.params = self.default_params
        else:
            self.update_params(params)
        self.par = True
        self.cat_encoder = None
        self.text_encoder = None
        self.col_dict = col_dict
        self.imputer = None
        self.feature_names = []

    def update_params(self, params):
        new_params = self.default_params
        for p in params.keys():
            temp_params = params[p]
            for pp in temp_params.keys():
                new_params[p][pp] = temp_params[pp]
        self.params = new_params

    def fit(self, data):
        '''

        :param data: pandas data frame containing all the columns listed in col_dict
        :return: none, encoders are saved within class
        '''
        col = 'text_cols'
        if col in self.col_dict.keys():
            print('fitting', col, ':', self.col_dict[col])
            self.text_encoder = TextFeature()
            self.text_encoder.par = self.par
            self.text_encoder.max_features = self.params[col]['max_features']
            self.text_encoder.min_freq = self.params[col]['min_freq']
            self.text_encoder.ngram_range = self.params[col]['ngram_range']
            self.text_encoder.min_len = self.params[col]['min_len']
            self.text_encoder.stop_words = self.params[col]['stop_words']
            self.text_encoder.fit(data[self.col_dict[col]])
            self.feature_names = self.feature_names + self.text_encoder.feature_names
        col = 'cat_cols'
        if col in self.col_dict.keys():
            print('fitting', col, ':', self.col_dict[col])
            self.cat_encoder = CatEncoder()
            self.cat_encoder.par = self.par
            self.cat_encoder.min_freq = self.params[col]['min_freq']
            self.cat_encoder.fit(data[self.col_dict[col]])
            self.feature_names = self.feature_names + self.cat_encoder.feature_names
        col = 'imputer_cols'
        if col in self.col_dict.keys():
            print('fitting', col, ':', self.col_dict[col])
            from sklearn.preprocessing import Imputer
            self.imputer = Imputer(strategy=self.params[col]['strategy'])
            self.imputer.fit(data[self.col_dict[col]])
            self.feature_names = self.feature_names + self.col_dict[col]
        col = 'zero_imputer_cols'
        if col in self.col_dict.keys():
            self.feature_names = self.feature_names + self.col_dict[col]

    def transform(self, data):
        '''

        :param data: a pandas data frame with all the columns listed in col_dict
        :return: scipy sparse matrix of features
        '''
        from scipy import sparse
        self.cat_encoder.par = self.text_encoder.par = self.par
        output_list = []
        col = 'text_cols'
        if col in self.col_dict.keys():
            output_list.append(self.text_encoder.transform(data[self.col_dict[col]]))
            print('transforming', col, ':', self.col_dict[col])
        col = 'cat_cols'
        if col in self.col_dict.keys():
            print('transforming', col, ':', self.col_dict[col])
            output_list.append(self.cat_encoder.transform(data[self.col_dict[col]]))
        col = 'imputer_cols'
        if col in self.col_dict.keys():
            print('transforming', col, ':', self.col_dict[col])
            output_list.append(sparse.csr_matrix(self.imputer.transform(data[self.col_dict[col]])))
        col = 'zero_imputer_cols'
        if col in self.col_dict.keys():
            import pandas as pd
            print('transforming', col, ':', self.col_dict[col])
            output_list.append(sparse.csr_matrix(data[self.col_dict[col]].fillna(0)))
        output = sparse.hstack(output_list)
        return output


