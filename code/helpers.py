from sentiment_analyzer import SentimentAnalyzer

def getEmotiDictionary(file_path = "../source/lexicon/EmoticonSentimentLexicon.txt"):
    emoti_dict = {}
    with open(file_path,"r") as trg:
        data = trg.read().split('\n')

    for i in range(len(data)):
        if "\t" in data[i]:
            key,value = data[i].split("\t")
            emoti_dict.setdefault(key,value)
    return emoti_dict

def getSetOfWordsFromFile(file_path = "../source/lexicon/NeutralitySigns.txt"):
    with open(file_path,"r") as trg:
        data = trg.read().split('\n')
    words = set(i for i in data if len(i) >=1 and i!="")
    return words

def getPolarValues(file_path):
    result_dict = {}
    with open(file_path,"r") as trg:
        data = trg.read().split("\n")
    for line in data:
        if "," in line:
            label,word = line.split(",")
            if len(word)>=1 and word!="":
                result_dict.setdefault(word,label)
    return result_dict

def initializeAnalyzer():
    path_to_emoti_file = "../source/lexicon/EmoticonSentimentLexicon.txt"
    path_to_neut_signs_file = "../source/lexicon/NeutralitySigns.txt"
    path_to_polar_nouns = "../source/lexicon/polarized_nouns.txt"
    path_to_polar_verbs = "../source/lexicon/polarazed_verbs.txt"
    path_to_polar_adjectives = "../source/lexicon/polarized_adjectives.txt"
    path_to_polar_conjunctions = "../source/lexicon/polarized_conjunctions.txt"
    path_to_foma_dividers = "../source/lexicon/foma_features_dividers.txt"
    path_to_foma_reversers = "../source/lexicon/foma_features_reversers.txt"
    path_to_meaning_reversers = "../source/lexicon/meaning_reverser.txt"
    path_to_punctuation_signs = "../source/lexicon/punctuation_signs.txt"

    emoti_dict = getEmotiDictionary(file_path=path_to_emoti_file)
    neut_set = getSetOfWordsFromFile(file_path=path_to_neut_signs_file)
    polar_noun = getPolarValues(file_path=path_to_polar_nouns)
    polar_vrb = getPolarValues(file_path=path_to_polar_verbs)
    polar_adj = getPolarValues(file_path=path_to_polar_adjectives)
    polar_conj = getPolarValues(file_path=path_to_polar_conjunctions)
    foma_dividers = getSetOfWordsFromFile(file_path=path_to_foma_dividers)
    foma_reversers = getSetOfWordsFromFile(file_path=path_to_foma_reversers)
    meaning_reversers = getSetOfWordsFromFile(file_path=path_to_meaning_reversers)
    punctuation_signs = getSetOfWordsFromFile(file_path=path_to_punctuation_signs)
    emoti_dict = organizedListOfEmoties(emoti_dict)
    analyzer = SentimentAnalyzer(
                                emoti_dict=emoti_dict,
                                neutral_signs=neut_set,
                                polar_nouns=polar_noun,
                                polar_verbs=polar_vrb,
                                polar_adjectives=polar_adj,
                                polar_conjunctions=polar_conj,
                                foma_dividers=foma_dividers,
                                foma_reversers=foma_reversers,
                                punctuation_signs=punctuation_signs,
                                meaning_reversers=meaning_reversers,
                                vrb_prob_coef=3,
                                sent_coef_decr=.2,
                                coef_of_postg_change=.1
                                )
    # print("FROM ANALYZER",analyzer.isPartOf("",analyzer.polar_noun))
    return analyzer

def organizedListOfEmoties(emoti_dict):
    temp_dict = {}
    for emot in emoti_dict:
        length = len(emot)
        if length in temp_dict:
            temp_dict[length].append((emot,emoti_dict[emot]))
        else:
            temp_dict[length] = [(emot,emoti_dict[emot])]
    max_length = max(list(temp_dict.keys()))
    result = []
    for i in range(max_length,0,-1):
        if i in temp_dict:
            for item in temp_dict[i]:
                result.append(item)
    return result




def main():
    print("LOG:\n\tDO NOT RUN THIS FILE!(This file is not supposed to be run you idiot)")
    pass

if __name__=="__main__":
    main()
