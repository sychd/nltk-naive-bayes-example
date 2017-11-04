class ClassifierConfiguration:
    __data_path = './data/'

    POS_FILENAME = __data_path + 'positive.json'
    NEG_FILENAME = __data_path + 'negative.json'
    TEST_FILENAME = __data_path + 'test.json'

    LANGUAGE = 'russian'
    POSITIVE_CLASS = 'positive'
    NEGATIVE_CLASS = 'negative'

    PREVIEW_TEXT_LENGTH = 30