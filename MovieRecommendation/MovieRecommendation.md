```python
import pandas as pd
```


```python
df = pd.read_csv("movies.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>10-12-2009</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>19-05-2007</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>245000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.sonypictures.com/movies/spectre/</td>
      <td>206647</td>
      <td>[{"id": 470, "name": "spy"}, {"id": 818, "name...</td>
      <td>en</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
      <td>107.376788</td>
      <td>[{"name": "Columbia Pictures", "id": 5}, {"nam...</td>
      <td>[{"iso_3166_1": "GB", "name": "United Kingdom"...</td>
      <td>26-10-2015</td>
      <td>880674609</td>
      <td>148.0</td>
      <td>[{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...</td>
      <td>Released</td>
      <td>A Plan No One Escapes</td>
      <td>Spectre</td>
      <td>6.3</td>
      <td>4466</td>
    </tr>
    <tr>
      <th>3</th>
      <td>250000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 80, "nam...</td>
      <td>http://www.thedarkknightrises.com/</td>
      <td>49026</td>
      <td>[{"id": 849, "name": "dc comics"}, {"id": 853,...</td>
      <td>en</td>
      <td>The Dark Knight Rises</td>
      <td>Following the death of District Attorney Harve...</td>
      <td>112.312950</td>
      <td>[{"name": "Legendary Pictures", "id": 923}, {"...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>16-07-2012</td>
      <td>1084939099</td>
      <td>165.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>The Legend Ends</td>
      <td>The Dark Knight Rises</td>
      <td>7.6</td>
      <td>9106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>260000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://movies.disney.com/john-carter</td>
      <td>49529</td>
      <td>[{"id": 818, "name": "based on novel"}, {"id":...</td>
      <td>en</td>
      <td>John Carter</td>
      <td>John Carter is a war-weary, former military ca...</td>
      <td>43.926995</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}]</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>07-03-2012</td>
      <td>284139100</td>
      <td>132.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>Lost in our world, found in another.</td>
      <td>John Carter</td>
      <td>6.1</td>
      <td>2124</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["genres"][0]
```




    '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'




```python
df["keywords"][0]
```




    '[{"id": 1463, "name": "culture clash"}, {"id": 2964, "name": "future"}, {"id": 3386, "name": "space war"}, {"id": 3388, "name": "space colony"}, {"id": 3679, "name": "society"}, {"id": 3801, "name": "space travel"}, {"id": 9685, "name": "futuristic"}, {"id": 9840, "name": "romance"}, {"id": 9882, "name": "space"}, {"id": 9951, "name": "alien"}, {"id": 10148, "name": "tribe"}, {"id": 10158, "name": "alien planet"}, {"id": 10987, "name": "cgi"}, {"id": 11399, "name": "marine"}, {"id": 13065, "name": "soldier"}, {"id": 14643, "name": "battle"}, {"id": 14720, "name": "love affair"}, {"id": 165431, "name": "anti war"}, {"id": 193554, "name": "power relations"}, {"id": 206690, "name": "mind and soul"}, {"id": 209714, "name": "3d"}]'



## clean text in genres and keywords


```python
import ast
```


```python
def convert(text) :
    l = []
    
    for i in ast.literal_eval(text) :
        l.append(i["name"])
    
    return l
```


```python
df["genres"] = df["genres"].apply(convert)
df["keywords"] = df["keywords"].apply(convert)
```


```python
df["genres"][0]
```




    ['Action', 'Adventure', 'Fantasy', 'Science Fiction']




```python
df["keywords"][0]
```




    ['culture clash',
     'future',
     'space war',
     'space colony',
     'society',
     'space travel',
     'futuristic',
     'romance',
     'space',
     'alien',
     'tribe',
     'alien planet',
     'cgi',
     'marine',
     'soldier',
     'battle',
     'love affair',
     'anti war',
     'power relations',
     'mind and soul',
     '3d']



## split text in overview column


```python
df["overview"][0]
```




    'In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.'




```python
df["overview"].isnull().sum()
```




    3




```python
df["overview"] = df["overview"].fillna("")
```


```python
df["overview"].isnull().sum()
```




    0




```python
df["overview"] = df["overview"].apply(lambda x: x.split())
```


```python
df["overview"][0] # text space is already handle in this column
```




    ['In',
     'the',
     '22nd',
     'century,',
     'a',
     'paraplegic',
     'Marine',
     'is',
     'dispatched',
     'to',
     'the',
     'moon',
     'Pandora',
     'on',
     'a',
     'unique',
     'mission,',
     'but',
     'becomes',
     'torn',
     'between',
     'following',
     'orders',
     'and',
     'protecting',
     'an',
     'alien',
     'civilization.']



## handle the text space


```python
df["genres"][0]
```




    ['Action', 'Adventure', 'Fantasy', 'Science Fiction']




```python
df["keywords"][0]
```




    ['culture clash',
     'future',
     'space war',
     'space colony',
     'society',
     'space travel',
     'futuristic',
     'romance',
     'space',
     'alien',
     'tribe',
     'alien planet',
     'cgi',
     'marine',
     'soldier',
     'battle',
     'love affair',
     'anti war',
     'power relations',
     'mind and soul',
     '3d']




```python
df["genres"] = df["genres"].apply(lambda x : [i.replace(" " , "") for i in x])
df["keywords"] = df["keywords"].apply(lambda x : [i.replace(" " , "") for i in x])
```


```python
df["genres"][0]
```




    ['Action', 'Adventure', 'Fantasy', 'ScienceFiction']




```python
df["keywords"][0]
```




    ['cultureclash',
     'future',
     'spacewar',
     'spacecolony',
     'society',
     'spacetravel',
     'futuristic',
     'romance',
     'space',
     'alien',
     'tribe',
     'alienplanet',
     'cgi',
     'marine',
     'soldier',
     'battle',
     'loveaffair',
     'antiwar',
     'powerrelations',
     'mindandsoul',
     '3d']



## create a column that stores all merged info


```python
df["info"] = df["overview"] + df["genres"] + df["keywords"]
```


```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>...</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>info</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000</td>
      <td>[Action, Adventure, Fantasy, ScienceFiction]</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[cultureclash, future, spacewar, spacecolony, ...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>[In, the, 22nd, century,, a, paraplegic, Marin...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>...</td>
      <td>10-12-2009</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
      <td>[In, the, 22nd, century,, a, paraplegic, Marin...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300000000</td>
      <td>[Adventure, Fantasy, Action]</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[ocean, drugabuse, exoticisland, eastindiatrad...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>[Captain, Barbossa,, long, believed, to, be, d...</td>
      <td>139.082615</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>...</td>
      <td>19-05-2007</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
      <td>[Captain, Barbossa,, long, believed, to, be, d...</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 21 columns</p>
</div>




```python
df["info"][0]
```




    ['In',
     'the',
     '22nd',
     'century,',
     'a',
     'paraplegic',
     'Marine',
     'is',
     'dispatched',
     'to',
     'the',
     'moon',
     'Pandora',
     'on',
     'a',
     'unique',
     'mission,',
     'but',
     'becomes',
     'torn',
     'between',
     'following',
     'orders',
     'and',
     'protecting',
     'an',
     'alien',
     'civilization.',
     'Action',
     'Adventure',
     'Fantasy',
     'ScienceFiction',
     'cultureclash',
     'future',
     'spacewar',
     'spacecolony',
     'society',
     'spacetravel',
     'futuristic',
     'romance',
     'space',
     'alien',
     'tribe',
     'alienplanet',
     'cgi',
     'marine',
     'soldier',
     'battle',
     'loveaffair',
     'antiwar',
     'powerrelations',
     'mindandsoul',
     '3d']




```python
df["info"] = df["info"].apply(lambda x : ' '.join(x))
```


```python
df["info"][0]
```




    'In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d'




```python
df["info"] = df["info"].apply(lambda x : x.lower())
```


```python
df["info"][0]
```




    'in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d'



## vectorization


```python
from sklearn.feature_extraction.text import CountVectorizer
```


```python
cv = CountVectorizer()
```


```python
vector = cv.fit_transform(df["info"]).toarray()
```


```python
vector[0]
```




    array([0, 0, 0, ..., 0, 0, 0], dtype=int64)



## porter stemmer


```python
from nltk.stem.porter import PorterStemmer
```


```python
ps = PorterStemmer()
```


```python
def stem(text) :
    y = []
    
    for i in text.split() :
        y.append(ps.stem(i))
        
    return " ".join(y)
```


```python
df["info"] = df["info"].apply(stem)
```


```python
df["info"][0]
```




    'in the 22nd century, a parapleg marin is dispatch to the moon pandora on a uniqu mission, but becom torn between follow order and protect an alien civilization. action adventur fantasi sciencefict cultureclash futur spacewar spacecoloni societi spacetravel futurist romanc space alien tribe alienplanet cgi marin soldier battl loveaffair antiwar powerrel mindandsoul 3d'



## cosine similarity


```python
# cosine similarity measures the simalarity between two vectors

from sklearn.metrics.pairwise import cosine_similarity
```


```python
cosine_similarity(vector)
```




    array([[1.        , 0.24455799, 0.20350679, ..., 0.22599838, 0.17342199,
            0.1278043 ],
           [0.24455799, 1.        , 0.29944476, ..., 0.25424067, 0.28203804,
            0.17155831],
           [0.20350679, 0.29944476, 1.        , ..., 0.28745128, 0.20954953,
            0.19217792],
           ...,
           [0.22599838, 0.25424067, 0.28745128, ..., 1.        , 0.32334299,
            0.1925571 ],
           [0.17342199, 0.28203804, 0.20954953, ..., 0.32334299, 1.        ,
            0.25734955],
           [0.1278043 , 0.17155831, 0.19217792, ..., 0.1925571 , 0.25734955,
            1.        ]])




```python
cs = cosine_similarity(vector)
```


```python
cs[0]
```




    array([1.        , 0.24455799, 0.20350679, ..., 0.22599838, 0.17342199,
           0.1278043 ])



## recommendation


```python
def recommendation(movie) :
    movie_index = df[df["title"] == movie].index[0]
    
    distance = cs[movie_index]
    movie_list = sorted(list(enumerate(distance)) , reverse= True , key= (lambda x : x[1]))[1:6]
    
    for i in movie_list :
        print(df["title"][i[0]])
```


```python
# recommendation("Avatar")
```


```python
user = input("Enter a movie name: ")
```

    Enter a movie name: Iron Man
    


```python
print("The recommendations are:\n")
recommendation(user)
```

    The recommendations are:
    
    Iron Man 3
    Captain America: Civil War
    Guardians of the Galaxy
    The Avengers
    Captain America: The First Avenger
    


```python

```
