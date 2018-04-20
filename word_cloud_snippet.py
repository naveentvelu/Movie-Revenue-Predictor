from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          max_font_size = 200,
                          background_color='white',
                          random_state=42,
                          collocations = False
                         ).generate(str(all_genres)) # all genres contains all the genres for the movie
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=1080)
