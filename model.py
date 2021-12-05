import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def content_based_recommendation(title,cs):
    try:
        index = (df[df["title"]==title].index.values[0])
    except IndexError:
        return
    
    similarity = cs[index]
    similarity_list = list(enumerate(similarity))
    similarity_list = sorted(similarity_list, key = lambda x: x[1]) 
    similarity_list = similarity_list[::-1]
    top_7 = [i[0] for i in similarity_list[1:8]]

    return (df['title'].iloc[top_7])


def random_recommendation(title):
    genre = df.loc[df['title'] == title, 'listed_in'].iloc[0]
    data = df[df['listed_in'] == genre]
    data.drop(data.index[data['title'] == title], inplace = True)
    if (data.shape[0]) > 7:
        data = (data.sample(7))
    return(data["title"])


df = pd.read_csv('netflix_titles.csv')

#EDA - Preprocessing
df = df[df['director'].notnull()]
df = df[df['cast'].notnull()]
df['rating'].fillna('PG-13',inplace = True)
df['duration'].fillna('90 min',inplace = True)
m = df['country'].mode()[0]
df['country'].fillna(m,inplace = True)

# First model is based on the description of a particular movie using cosine similarity and bag of words.
# Word vector will be calculated using TF-IDF

tfidf = TfidfVectorizer(stop_words='english')
df["description"].fillna('')
word_matrix = tfidf.fit_transform(df['description'])

#print(word_matrix.shape)

#print(tfidf.get_feature_names_out()[3000:3008])

cs = cosine_similarity(word_matrix, word_matrix) #calculate cosine simality between movies based on description
#print(cs.shape,"\n",cs[0])

top_7_recommendations = content_based_recommendation("Ganglands",cs)
print("Top recommendations based on the plot:")
print(top_7_recommendations)
top_7_random_recommendations = random_recommendation("Ganglands")
print("Top recommendations based on the genre:")
print(top_7_random_recommendations)

def processData(x):
    if len(x.split(",")) > 3:
        return x.split(",")[0:3]
    else:
        return x.split(",")

def processing(x):
    if type(x) == list:
            return [i.replace(" ", "").lower() for i in x]
    else:
            return (x.replace(" ", "").lower())

def create_soup(data):
    return  ' ' + ' '.join(data['cast']) + ' ' + data['director'] + ' ' + ' '.join(data['listed_in'])

def recommendation_director(title):
    data = df
    data["cast"] = data["cast"].apply(processData)
    data["listed_in"] = data["listed_in"].apply(processData)
    data["director"] = data["director"].apply(lambda x: x.split(",")[0])
    
    categories = ["director","listed_in","cast"]

    for c in categories:
        data[c] = data[c].apply(processing)

    data["data_soup"] = data.apply(create_soup, axis=1)
    
    count = CountVectorizer(stop_words='english')
    cm = count.fit_transform(data['data_soup'])

    #Use cosine similarity 
    cs = cosine_similarity(cm, cm)
    res = content_based_recommendation(title,cs)
    return res

top_7_director = recommendation_director("Ganglands")
print("Top Movies based on combination of Cast, director and Genre:")
print(top_7_director)
