import os
import numpy as np

class SentimentAnalyzer:

    def __init__(
                    self,
                    emoti_dict,
                    neutral_signs,
                    polar_nouns,
                    polar_verbs,
                    polar_adjectives,
                    polar_conjunctions,
                    foma_dividers,
                    foma_reversers,
                    punctuation_signs,
                    meaning_reversers,
                    vrb_prob_coef,
                    sent_coef_decr,
                    coef_of_postg_change
                ):

        self.emoti_dict = emoti_dict
        self.neutrality_signs = neutral_signs
        self.polar_noun = polar_nouns
        self.polar_vrb = polar_verbs
        self.polar_adj = polar_adjectives
        self.polar_conjunctions = polar_conjunctions
        self.foma_features_dividers = foma_dividers
        self.foma_features_reversers = foma_reversers
        self.punctuation_signs = punctuation_signs
        self.meaning_reversers = meaning_reversers
        self.vrb_prob_coef = vrb_prob_coef
        self.per_sentencetone_coef_decrementer = sent_coef_decr
        self.coef_of_postg_change = coef_of_postg_change

    def calculateSentimentFromTonalityResults(self,tn,tv,ta):
        tot_tone = 0
        if tv >= 1:
            if tn>=1: tot_tone = 1
            elif tn<=-1: tot_tone = -.5
            else: tot_tone = 1
        elif tv <= -1:
            if tn>=1: tot_tone = -.5
            elif tn<=-1: tot_tone = .5
            else: tot_tone = -1
        else:
            if tn>=0:
                tot_tone = .5
            else:
                tot_tone = -.5

        adj_sign = 1 if ta >=0 else -1

        adj_coef = 0
        if ta != 0:
            adj_coef = (.75-(1/(2**abs(ta)))) * adj_sign
        tot_tone+=adj_coef
        return tot_tone

    def checkForEmoti(self,sentence):
        result = 0
        for key,value in self.emoti_dict:
            if key in sentence:
                result += int(value)*sentence.count(key)
                sentence = sentence.replace(key," ")
        bracket_total = self.countTotalBrackets(sentence)
        sentence = sentence.replace("(",",")
        sentence = sentence.replace(")",",")
        result += bracket_total
        self.trg_sentence = sentence
        if result==0:
            return 0
        elif result>0:
            return 1
        else:
            return -1

    def getTokenizedSentence(self, sentence):
        with open("./../source/temp/temp_input_storage.txt","w") as in_trg:
            in_trg.write(sentence)
        command_to_execute =  """cat < ./../source/temp/temp_input_storage.txt  | flookup -i -x -w "" ./../source/rules/token.bin > ./../source/temp/temp_output_storage.txt"""
        os.system(command_to_execute)
        with open("./../source/temp/temp_output_storage.txt","r") as out_trg:
            data = out_trg.read()
        return [i for i in data.split("\n") if len(i)>0]
        # return data1

    def countTotalBrackets(self, sentence):
        total,step = 0,0
        for i in range(len(sentence)):
            if sentence[i] == "(":
                if step > 0:
                    total +=step
                    step=0
                step -=1
            elif sentence[i] == ")":
                if step < 0:
                    total +=step
                    step=0
                step +=1
        total+=step
        return total

    def checkForNeutralitySigns(self, tokens):
        if any(i in tokens for i in self.neutrality_signs):
            return 1
        return 0

    def removeProperNames(self):
        list_of_words = self.trg_sentence.split(" ")
        if len(list_of_words)<2:
            return list_of_words
        first_word = list_of_words[0].lower()
        result_list = [first_word]
        for i in range(1,len(list_of_words)):
            word = list_of_words[i]
            # TODO: While this chunk is responsible for removing CAPSWORDS you need to learn how to deal with ->WHO<-(abbriviations)
            # if all(ltr==ltr.upper() for ltr in word):
            #     result_list.append(word.lower())
            if any(ltr==ltr.upper() for ltr in word if not ltr in self.punctuation_signs):
                pass
            else:
                result_list.append(word)
        self.trg_sentence = " ".join(i for i in result_list)

    def getFomaOutputOfWord(self, word):
        with open("./../source/temp/foma_input_storage.txt","w") as in_trg:
            in_trg.write(word)
        command_to_execute =  """cat < ./../source/temp/foma_input_storage.txt  | flookup  -x -w "" ./../source/rules/kazakh.bin > ./../source/temp/foma_output_storage.txt"""
        os.system(command_to_execute)
        with open("./../source/temp/foma_output_storage.txt","r") as out_trg:
            foma_output = out_trg.read()
        return foma_output

    def isPartOf(self, trg_word, trg_set):
        if len(trg_word)<=1:
            return 0
        if trg_word in trg_set:
            # print("I KNEW IT")
            return trg_word
        for elem in trg_set:
            if elem.find(trg_word)==0 or trg_word.find(elem)==0:
                return elem
        return False

    def isWordIsPartOf(self, trg_word, trg_set):
        if any(trg_word == word for word in trg_set):
            return trg_word
        return False

    def divideToSubSentences(self,list_of_words, dict_of_foma_results):
        sub_sent_list = []
        prediction_list = [0]
        temp_list = []
        for word in list_of_words[::-1]:
            if word in self.punctuation_signs:
                if temp_list:
                    sub_sent_list.append(temp_list)
                    prediction_list.append(0)
                    temp_list = []
                continue
            partly_conj = self.isPartOf(word, self.polar_conjunctions)
            if partly_conj:
                if temp_list:
                    sub_sent_list.append(temp_list)
                    prediction_list.append(self.polar_conjunctions[partly_conj])
                    temp_list = []
                continue
            if any(rules in dict_of_foma_results[word] for rules in self.foma_features_dividers):
                if temp_list:
                    sub_sent_list.append(temp_list)
                    prediction_list.append(0)
                temp_list = [word]
                continue
            temp_list.append(word)
        if temp_list:
            sub_sent_list.append(temp_list)

        return sub_sent_list, prediction_list

    def calculatePosTag(self, prob_list, count_list):
        total_list = prob_list*count_list
        index_of_max = total_list.argmax()
        if index_of_max==0:
            return "Noun"
        elif index_of_max==1:
            return "Verb"
        elif index_of_max==2:
            return "Adj"

    def updatePosTagList(self,pos_tag_list, pos_tag):
        d = self.coef_of_postg_change
        if pos_tag == "Noun":
            return pos_tag_list+np.array([-d/2,-d/2,d])
        if pos_tag == "Verb":
            return pos_tag_list+np.array([d,-d/2,-d/2])
        if pos_tag == "Adj":
            return pos_tag_list+np.array([-d/2,-d/2,d])

    def performFomaAnlysisOnSentences(self, sub_sentences, dict_of_foma_results):
        result_tonality_list = []
        coef_of_reverse = 1

        for sentence in sub_sentences:

            tone_adj = 0
            tone_verb = 0
            tone_noun = 0
            verb_prob_delta = -1*self.vrb_prob_coef
            pos_tag_prob_list = np.array([.3,.5,.2]) #TODO: TUNE IT!!!
            for word in sentence:
                root=""

                if word in self.meaning_reversers:
                    coef_of_reverse = -1
                    continue

                foma_result_lines = [line for line in dict_of_foma_results[word].split("\n") if len(line) > 1]
                foma_second_col_features = [i.split("+")[1][0:3] for i in foma_result_lines]
                pos_tag_cnted_list = np.array([
                                    foma_second_col_features.count("Nou"),
                                    foma_second_col_features.count("Ver"),
                                    foma_second_col_features.count("Adj")
                                    ])

                pos_tag = self.calculatePosTag(pos_tag_prob_list,pos_tag_cnted_list)
                pos_tag_prob_list = self.updatePosTagList(pos_tag_prob_list, pos_tag)
                try:
                    pos_feature_line = foma_result_lines[foma_second_col_features.index(pos_tag[:3])]
                except:
                    pos_feature_line = ""


                root = foma_result_lines[0].split("+")[0]
                if len(root)<=1:
                    root = word

                if pos_tag == "Verb":
                    partly_verb = self.isPartOf(root,self.polar_vrb)
                    negative_mult = -1 if self.isPartOf("Neg",pos_feature_line) else 1
                    if partly_verb:
                        if tone_verb==0:
                            tone_verb = int(self.polar_vrb.get(partly_verb))
                        else:
                            tone_verb *= int(self.polar_vrb.get(partly_verb))

                    if negative_mult==-1:
                        tone_verb*=negative_mult
                        negative_mult=1
                    if coef_of_reverse==-1:
                        tone_verb*=coef_of_reverse
                        coef_of_reverse=1

                elif pos_tag[0:3] == "Adj":
                    partly_adj = self.isPartOf(root,self.polar_adj)
                    if partly_adj:

                        tone_adj += (int(self.polar_adj.get(partly_adj)) * coef_of_reverse)
                    coef_of_reverse=1
                else:
                    partly_noun = self.isPartOf(root,self.polar_noun)
                    if partly_noun:
                        if tone_noun==0:
                            tone_noun = int(self.polar_noun.get(partly_noun))
                        else:
                            tone_noun*= int(self.polar_noun.get(partly_noun))
                    tone_noun*=coef_of_reverse
                    coef_of_reverse=1
            total_tone = self.calculateSentimentFromTonalityResults(tone_noun, tone_verb, tone_adj)
            result_tonality_list.append(total_tone)

        return result_tonality_list

    def checkForTonality(self, list_of_words):
        dict_of_foma_results = {}
        for word in list_of_words:
            if len(word)>0:
                foma_representation = self.getFomaOutputOfWord(word)
                dict_of_foma_results.setdefault(word, foma_representation)

        sub_sentences_list,prediction_list = self.divideToSubSentences(list_of_words,dict_of_foma_results)
        result_tonality_list = self.performFomaAnlysisOnSentences(sub_sentences_list,dict_of_foma_results)
        #TODO: write a normal formula for calculation overall tone
        # for line in sub_sentences_list:
        #     print(line)
        # print(*result_tonality_list)
        total_result = self.sumOfSentencesResults(result_tonality_list)
        if abs(total_result)*4<=1:
            return 0
        return total_result

    def sumOfSentencesResults(self, result_list):
        coef = 1.0
        total = 0
        for sentence_tone_res in result_list:
            total+= sentence_tone_res * coef
            coef = max(.3,coef-self.per_sentencetone_coef_decrementer)
        return total

    def getConjunctionType(self, word):
        if word in self.polar_conj:
            return self.polar_conj.get(word)
        return 0

    def analyze(self,sentence):

        self.trg_sentence = sentence

        emoti_check = self.checkForEmoti(sentence)
        if emoti_check!=0:
            return emoti_check

        self.removeProperNames()

        tokenized_sentence = self.getTokenizedSentence(self.trg_sentence)
        neutrality_check = self.checkForNeutralitySigns(tokenized_sentence)

        if neutrality_check!=0:
            return 0

        tonality = self.checkForTonality(tokenized_sentence)
        return tonality
