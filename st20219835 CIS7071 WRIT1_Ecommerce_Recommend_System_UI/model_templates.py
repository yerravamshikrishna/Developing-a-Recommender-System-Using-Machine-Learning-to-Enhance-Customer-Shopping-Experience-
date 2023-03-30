# recommendation for
# create engine for the recommendation

class ContentBasedRecommendorSystem:
    """
    A ENGINE FOR WALMART PRODUCT RECOMENTATION
    @process --> getVectors --> dimension reduction --> cosine similarity --> get indexes or recommented products
    """
    def __init__(self, vect_type ='count', n_dim=50):
        """
        Constructor..
        Initialises vectorizer .
        The vectorizer removes english stopwords from the word that were not removed
        """
        print("****** SYSYTEM ENGINE INITIATED *******\n")
        #define vectorizer

        self.vect_type = vect_type
        self.n_dim = n_dim

        #CHECK THE VECTORIZER BEING USED.
        if self.vect_type =='count':
            print("Using Count Vectorizer as Vectorizer")
            self.vect = CountVectorizer(encoding = 'utf-8', stop_words="english" ,analyzer='word')
        else:
            print("Using TFIDF Vectorizer as Vectorizer")
            self.vect = TfidfVectorizer(analyzer = 'word',stop_words="english", encoding = 'utf-8')
    

    def fit(self,df, column):
        """
        This function performs main process of the engine.
        Creates a model and return  similarity indexes or the recommended
        """
        self.df =df
        #get the index as a product id
        if "Uniq Id" in df.columns:
            self.df.set_index("Uniq Id", inplace=True)
        else:
            pass
        # fit 
        data = df[f'{column}']
        # fill nulls with a value.

        #fill all nulls with missing product keyword incase it is not present
        data.fillna("This product does not have any descriptions" , inplace = True)
        
        #convert the vectorizer
        self.vect.fit(data.values)
        # transform the data
        self.tf_df = self.vect.transform(data.values)
        #reduce the matrix
        self.reduced_matrix = self.dimension_reduction(self.tf_df)
        print("Done wit dimension reduction")
    
        # Compute the cosine similarity matrix
        self.cos_similarity  = cosine_similarity(self.reduced_matrix , self.reduced_matrix)
        
        # create a new series for holding product id and index in small case
        self.recomender_indices = pd.Series(df.index)
    
    def dimension_reduction(self , sparse_matrix):
        """
        Perfoms Dimension reduction by reducing the prodcut matrix dimension to 100 by default.
        """
        print("Started Dimension Reduction")
        trancator = TruncatedSVD(n_components = self.n_dim)
        return trancator.fit_transform(sparse_matrix).astype('f')
        			

    def Weight_Rating_score(self ,data):
        """
        This functions is used incase title provided by useris not found in the available database.
        It uses averaging of the rating score and return top 15% product
        """
        #get the percentile for top recommentation
        x_percent = data['Num Of Reviews'].quantile(0.85)
        #mean
        MEAN = data['Average Rating'].mean()
        votes = data['Number Of Ratings']
        vote_avg = data['Average Rating']
        # votes based recommentation
        return (votes/(votes+1+x_percent) * vote_avg) + (x_percent/(x_percent+votes+1) * MEAN)
        

    def predict(self , product_id , numbers=5):
        """
        Function to return recomended products
        """
        print("****** THE FOLLOWING ARE RECOMMENDATIONS **********")
        
        # a list to hold the product to be recommended
        recommended_products = []
        similarity_scores = []
        try:
            #get index of the product that matches the id given
            idx = self.recomender_indices[self.recomender_indices == product_id].index[0]

            #find highest cosine_sim this product id shares with other product ids extracted earlier and save it in a Series
            score_series = pd.Series(self.cos_similarity[idx]).sort_values(ascending = False)

            #get indexes of the 'n' most similar products
            top_n_indexes = list(score_series.iloc[1:numbers + 1].index)

            #populating the list with product ids of n matching product
            for i in top_n_indexes:
                recommended_products.append(list(df.index)[i])
                similarity_scores.append(score_series[i])
        except Exception as e:
            print("error Occured recommending based on popularity")
            #create a new feature score
            new_df = self.df.copy()
            new_df['score'] = self.Weight_Rating_score(new_df)

            #get recommended products randomly from the top 100 ranked products
            return [0 for _ in range(numbers)], new_df.sort_values(by= 'score', ascending=False)["score"][:1000].sample(numbers).index.tolist()
            
        
        return similarity_scores, recommended_products
    
    
    
# recommendation for
# create engine for the recommendation

class NearestNeighboursContentBasedRecommendorSystem:
    """
    A ENGINE FOR WALMART PRODUCT RECOMENTATION
    @process --> getVectors --> dimension reduction --> cosine similarity --> get indexes or recommented products
    """
    def __init__(self, vect_type ='tfidf', n_dim=100):
        """
        Constructor..
        Initialises vectorizer .
        The vectorizer removes english stopwords from the word that were not removed
        """
        print("****** SYSYTEM ENGINE INITIATED USING NEAREST NEIGHBOURS ALGORITHM*******\n")
        #define vectorizer

        self.vect_type = vect_type
        self.n_dim = n_dim

        #CHECK THE VECTORIZER BEING USED.
        if self.vect_type =='count':
            print("Using Count Vectorizer as Vectorizer")
            self.vect = CountVectorizer(encoding = 'utf-8', stop_words="english" ,analyzer='word')
        else:
            print("Using TFIDF Vectorizer as Vectorizer")
            self.vect = TfidfVectorizer(analyzer = 'word',stop_words="english", encoding = 'utf-8')
    

    def fit(self,df, column):
        """
        This function performs main process of the engine.
        Creates a model and return  similarity indexes or the recommended
        """
        self.df =df
        #get the index as a product id
        if "Uniq Id" in df.columns:
            self.df.set_index("Uniq Id", inplace=True)
        else:
            pass
        # fit 
        data = df[f'{column}']
        # fill nulls with a value.

        #fill all nulls with missing product keyword incase it is not present
        data.fillna("This product does not have any descriptions" , inplace = True)
        
        #convert the vectorizer
        self.vect.fit(data.values)
        
        # transform the data
        self.tf_df = self.vect.transform(data.values)
        #reduce the matrix
        self.reduced_matrix = self.dimension_reduction(self.tf_df)
        print("Done wit dimension reduction")

        # Using NearestNeighbors model and kneighbors() method to find k neighbors.
        # Setting n_neighbors = 5 to find 5 similar products 
        # Using auto algorthm to decide on the best algorith to be used for getting k neighbours
        self.neigh_on_count_vec = NearestNeighbors(n_neighbors=5, algorithm='auto')

        # train the algorithm
        self.neigh_on_count_vec.fit(self.reduced_matrix)

        # create a new series for holding product id and index in small case
        self.recomender_indices = pd.Series(df.index)
    
    def dimension_reduction(self , sparse_matrix):
        """
        Perfoms Dimension reduction by reducing the prodcut matrix dimension to 100 by default.
        """
        print("Started Dimension Reduction")
        self.trancator = TruncatedSVD(n_components = self.n_dim)
        return self.trancator.fit_transform(sparse_matrix).astype('f')
        			

    def Weight_Rating_score(self ,data):
        """
        This functions is used incase title provided by useris not found in the available database.
        It uses averaging of the rating score and return top 15% product
        """
        #get the percentile for top recommentation
        x_percent = data['Num Of Reviews'].quantile(0.85)
        #mean
        MEAN = data['Average Rating'].mean()
        votes = data['Number Of Ratings']
        vote_avg = data['Average Rating']
        # votes based recommentation
        return (votes/(votes+x_percent) * vote_avg) + (x_percent/(x_percent+votes) * MEAN)
        

    def predict(self , product_id , numbers=5):
        """
        Function to return recomended products
        """
        print("****** THE FOLLOWING ARE RECOMMENDATIONS USING NEAREST NEIGHBOURS ALGORTHM **********")
        
        try:
            # get the product detrails with the id given
            product_desc = self.df.loc[product_id]['clean_text']
            # transform the above into a vector
            prod_mtx = self.vect.transform([product_desc])
            # reduce the dimension using the trucated svd object trained above
            prod_mtx_reduced = self.trancator.transform(prod_mtx).astype("f")
            # get recommende product indexes
            distances, indices_preds = self.neigh_on_count_vec.kneighbors(prod_mtx_reduced)
            # get the product IDs of the recommended product.
            return  1 / (1 + distances), self.df.index.values[list(indices_preds[0])]


        except Exception as e:
            print("error Occured recommending based on popularity")
            #create a new feature score
            new_df = df.copy()
            new_df['score'] = self.Weight_Rating_score(new_df)

            #get recommended products randomly from the top 100 ranked products
            return [0 for _ in range(numbers) ], new_df.sort_values(by= 'score', ascending=False)["score"][:1000].sample(numbers).index.tolist()