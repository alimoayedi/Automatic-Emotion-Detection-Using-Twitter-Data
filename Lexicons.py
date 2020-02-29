class Lexicons:
    def __init__(self):
        self.angerLexiconURL_list = [
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC Affect Intensity Lexicon-anger Lexicon.txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Hashtag-Emotion-Lexicon-v0.2-anger lexicon.txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emotion-Lexicon-Wordlevel-v0.92-anger lexicon.txt", "binary"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Hashtag-Sentiment-Lexicon-v1.0.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Hashtag-Sentiment-AffLexNegLex-v1.0.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emoticon-Lexicon-v1.0.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Emoticon-AFFLEX-NEGLEX-unigrams.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emoticon-Lexicon-v1.0-bigram.txt", "biword"],
            # scores using following lexicon already extracted and saved a csv file under name of "Lexicon Nine"
            # [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emoticon-Lexicon-v1.0-pairgram.txt","pairword"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\SentiWordNet3.0.txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Ratings_Warriner(Valence).txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Ratings_Warriner(Arousal).txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Ratings_Warriner(Dominance).txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\BingLiu-unigram-mixed.txt", "binary"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\BingLiu-bigram-mixed.txt", "biword"]
        ]
        self.joyLexiconURL_list = [
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC Affect Intensity Lexicon-joy Lexicon.txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Hashtag-Emotion-Lexicon-v0.2-joy lexicon.txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emotion-Lexicon-Wordlevel-v0.92-joy lexicon.txt", "binary"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Hashtag-Sentiment-Lexicon-v1.0.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Hashtag-Sentiment-AffLexNegLex-v1.0.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emoticon-Lexicon-v1.0.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Emoticon-AFFLEX-NEGLEX-unigrams.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emoticon-Lexicon-v1.0-bigram.txt", "biword"],
            # scores using following lexicon already extracted and saved a csv file under name of "Lexicon Nine"
            # [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emoticon-Lexicon-v1.0-pairgram.txt","pairword"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\SentiWordNet3.0.txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Ratings_Warriner(Valence).txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Ratings_Warriner(Arousal).txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Ratings_Warriner(Dominance).txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\BingLiu-unigram-mixed.txt", "binary"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\BingLiu-bigram-mixed.txt", "biword"]
        ]

        self.fearLexiconURL_list = [
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC Affect Intensity Lexicon-fear Lexicon.txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Hashtag-Emotion-Lexicon-v0.2-fear lexicon.txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emotion-Lexicon-Wordlevel-v0.92-fear lexicon.txt", "binary"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Hashtag-Sentiment-Lexicon-v1.0.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Hashtag-Sentiment-AffLexNegLex-v1.0.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emoticon-Lexicon-v1.0.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Emoticon-AFFLEX-NEGLEX-unigrams.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emoticon-Lexicon-v1.0-bigram.txt", "biword"],
            # scores using following lexicon already extracted and saved a csv file under name of "Lexicon Nine"
            # [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emoticon-Lexicon-v1.0-pairgram.txt","pairword"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\SentiWordNet3.0.txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Ratings_Warriner(Valence).txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Ratings_Warriner(Arousal).txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Ratings_Warriner(Dominance).txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\BingLiu-unigram-mixed.txt", "binary"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\BingLiu-bigram-mixed.txt", "biword"]
        ]

        self.sadnessLexiconURL_list = [
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC Affect Intensity Lexicon-sadness Lexicon.txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Hashtag-Emotion-Lexicon-v0.2-sadness lexicon.txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emotion-Lexicon-Wordlevel-v0.92-sadness lexicon.txt", "binary"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Hashtag-Sentiment-Lexicon-v1.0.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Hashtag-Sentiment-AffLexNegLex-v1.0.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emoticon-Lexicon-v1.0.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Emoticon-AFFLEX-NEGLEX-unigrams.txt", "psweighted"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emoticon-Lexicon-v1.0-bigram.txt", "biword"],
            # scores using following lexicon already extracted and saved a csv file under name of "Lexicon Nine"
            # [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\NRC-Emoticon-Lexicon-v1.0-pairgram.txt","pairword"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\SentiWordNet3.0.txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Ratings_Warriner(Valence).txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Ratings_Warriner(Arousal).txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\Ratings_Warriner(Dominance).txt", "decimal"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\BingLiu-unigram-mixed.txt", "binary"],
            [r"D:\Thesis\Thesis-CE\Phyton Program\lexicons\BingLiu-bigram-mixed.txt", "biword"]
        ]

    def getLexicons(self, emotion: str):
        if emotion == 'anger':
            return self.angerLexiconURL_list
        elif emotion == 'joy':
            return self.joyLexiconURL_list
        elif emotion == 'fear':
            return self.fearLexiconURL_list
        else:
            return self.sadnessLexiconURL_list
