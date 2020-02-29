class TestTrainFiles:
    def __init__(self):
        
        self.testFile_list = [
            # [r"D:\Thesis\Thesis-CE\Phyton Program\TweetToken-02-Data\test-edited\2018-EI-oc-En-anger-test-gold.txt", "anger"]
            # [r"D:\Thesis\Thesis-CE\Phyton Program\TweetToken-02-Data\test-edited\2018-EI-oc-En-joy-test-gold.txt", "joy"]
            # [r"D:\Thesis\Thesis-CE\Phyton Program\TweetToken-02-Data\test-edited\2018-EI-oc-En-fear-test-gold.txt", "fear"]
            # [r"D:\Thesis\Thesis-CE\Phyton Program\TweetToken-02-Data\test-edited\2018-EI-oc-En-sadness-test-gold.txt", "sadness"]
            ]
        
        self.devFile_list = [
            ["D:\Thesis\Thesis-CE\Phyton Program\TweetToken-02-Data\develope-edited\2018-EI-oc-En-anger-dev.txt", "anger"]
            # [r"D:\Thesis\Thesis-CE\Phyton Program\TweetToken-02-Data\develope-edited\2018-EI-oc-En-joy-dev.txt", "joy"]
            # [r"D:\Thesis\Thesis-CE\Phyton Program\TweetToken-02-Data\develope-edited\2018-EI-oc-En-fear-dev.txt", "fear"]
            # [r"D:\Thesis\Thesis-CE\Phyton Program\TweetToken-02-Data\develope-edited\2018-EI-oc-En-sadness-dev.txt", "sadness"]
            ]

        self.trainFile_list = [
            [r"D:\Thesis\Thesis-CE\Phyton Program\TweetToken-02-Data\train-edited\EI-oc-En-anger-train.txt", 'anger']
            # [r"D:\Thesis\Thesis-CE\Phyton Program\TweetToken-02-Data\train-edited\EI-oc-En-joy-train.txt", 'joy']
            # [r"D:\Thesis\Thesis-CE\Phyton Program\TweetToken-02-Data\train-edited\EI-oc-En-fear-train.txt", "fear"]
            # [r"D:\Thesis\Thesis-CE\Phyton Program\TweetToken-02-Data\train-edited\EI-oc-En-sadness-train.txt", "sadness"]
            ]

    def getFile(self, file: str):
        if file == 'train':
            return self.trainFile_list
        elif file == 'develop':
            return self.devFile_list
        elif file == 'test':
            return self.testFile_list
