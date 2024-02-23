import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt

text = """The moon cast a silvery glow over the tranquil lake, its reflection shimmering on the water's surface like scattered diamonds. Sarah sat at the edge of the dock, her thoughts drifting like the gentle ripples below. She pondered the mysteries of the universe, lost in the symphony of night sounds that surrounded her. In the distance, a lone owl hooted mournfully, adding to the nocturnal chorus. It was a moment suspended in time, where the boundaries between reality and dreams blurred into obscurity."""

tokenized_words = word_tokenize(text)
tokenized_sentence = sent_tokenize(text)
#print(word_tokenize(text))
#print(sent_tokenize(text))


from nltk.probability import FreqDist
fd= FreqDist(tokenized_words)
print(fd)
print(fd.most_common(3)) #가장 많이 나온 3가지
fd.plot(30, cumulative=False)
plt.show()



#nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
#print(stopwords)  #불용어 tokenize 할 때 쉽게 제거가능하게 만들어둠 

tagged_tokens = nltk.pos_tag(tokenized_words)
#print(tagged_tokens)

from nltk.corpus import wordnet
synonyms = []
a=[]
for syn in wordnet.synsets("sad"):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
        a.append(lemma)

from nltk.tag import brill, brill_trainer
train_data = [('surrounded', 'VBD'), ('her', 'PRP'), ('.', '.'), ('In', 'IN'), ('the', 'DT'), ('distance', 'NN'), (',', ','), ('a', 'DT'), ('lone', 'NN'), ('owl', 'NN'), ('hooted', 'VBD'), ('mournfully', 'RB'), (',', ','), ('adding', 'VBG'), ('to', 'TO'), ('the', 'DT'), ('nocturnal', 'JJ'), ('chorus', 'NN'), ('.', '.'), ('It', 'PRP'), ('was', 'VBD'), ('a', 'DT'), ('moment', 'NN'), ('suspended', 'VBN'), ('in', 'IN'), ('time', 'NN'), (',', ','), ('where', 'WRB'), ('the', 'DT'), ('boundaries', 'NNS'), ('between', 'IN'), ('reality', 'NN'), ('and', 'CC'), ('dreams', 'NNS'), ('blurred', 'VBD'), ('into', 'IN'), ('obscurity', 'NN'), ('.', '.')]

#태그된 문장 데이티ㅓ 생성
tagged_sentences = [sentence for sentence in train_data]
