# -*- coding: utf-8 -*-
# Coreference Resolution
# Final Project
# Bogdan Kostić, 785726, 19 March 2019
# Python 3.7.2

import sys
import os
from collections import deque
from nltk.corpus import wordnet
import csv

import OntonotesParser
import MentionPair

# The FeatureExtractor class iterates over OntoNotes files provided by OntonotesParser and extracts
# 12 features out of mention-pairs.
class FeatureExtractor :

  def __init__(self,input_data) :

    # read the gender data from external resource (Bergsma & Lin 2006)
    self.gender_frequencies = dict()
    gender_file = open('resources/gender.data','r')
    gender_data = csv.reader(gender_file,delimiter='\t')
    for entry in gender_data :
      key = entry[0]
      gender_counts = list(map(int,entry[1].split()))[0:3]
      self.gender_frequencies[key] = gender_counts
    gender_file.close()

    parser = OntonotesParser.Ontonotes()

    if os.path.isdir(input_data) :
      # data_iterator is a generator that iterates over every sentence in each file of each (sub)folder
      self.data_iterator = parser.dataset_iterator(input_data)
    elif os.path.isfile(input_data) :
      # data_iterator is a generator that iterates over every sentence in the file
      self.data_iterator = parser.sentence_iterator(input_data)
    else :
      print("ERROR: Cannot process input data", file=sys.stderr)
      print("'{}' does not lead to a file or directory".format(input_data), file=sys.stderr)
      sys.exit(1)

  # yields a subset of the training instances for reasons of memory limitations  
  def training_instances_iterator(self,docs_per_step=10) :

    current_number_of_documents = 0
    training_instances = []

    current_document = []
    current_document_id = ''
    current_part_number = -1
    current_sentence_idx = 0
    extracted_nps = []

    for sentence in self.data_iterator :
      # sentences are added to current_document, if they pertain to the same document
      if sentence.document_id == current_document_id and sentence.part_number == current_part_number :
        current_document.append(sentence)

        current_sentence_idx += 1

        # extract NPs in current sentence
        extracted_nps.extend(self._extract_nps(sentence.parse_tree,current_sentence_idx))

      # end of current_document, so we can extract features of this document
      # + start of new document, so document_id and part_number are updated
      else :
        training_instances = self._extract_mention_pairs_with_features(extracted_nps,current_document)

        if current_number_of_documents == docs_per_step :
          current_number_of_documents = 0
          yield training_instances

        # starting new document
        current_number_of_documents += 1
        current_document_id = sentence.document_id
        current_part_number = sentence.part_number
        current_sentence_idx = 0
        current_document = [sentence]
        extracted_nps = self._extract_nps(sentence.parse_tree,current_sentence_idx)

    yield training_instances

  # yields a document with its extracted NPs + features (used to predict coreference chains per doc)
  def document_iterator(self) :
    current_document = []
    current_document_id = ''
    current_part_number = -1
    current_sentence_idx = 0
    extracted_nps = []

    for sentence in self.data_iterator :
      # sentences are added to current_document, if they pertain to the same document
      if sentence.document_id == current_document_id and sentence.part_number == current_part_number :
        current_document.append(sentence)

        current_sentence_idx += 1

        # extract NPs in current sentence
        extracted_nps.extend(self._extract_nps(sentence.parse_tree,current_sentence_idx))

      # end of current_document, so we can yield the document an its mentions
      # + start of new document, so document_id and part_number are updated
      else :
        yield current_document, extracted_nps

        # starting a new document
        current_document_id = sentence.document_id
        current_part_number = sentence.part_number
        current_sentence_idx = 0
        current_document =[sentence]
        extracted_nps = self._extract_nps(sentence.parse_tree,current_sentence_idx)

  def extract_feature_vector(self,antecedent,anaphora,document) :
    antecedent_sentence = document[antecedent[0]]
    anaphora_sentence = document[anaphora[0]]

    # independent features
    antecedent = self._extract_independent_features(antecedent,antecedent_sentence)
    anaphora = self._extract_independent_features(anaphora,anaphora_sentence)

    # pair features
    mention_pair = MentionPair.MentionPair(self._extract_features(antecedent,
                                                                  anaphora,
                                                                  antecedent_sentence,
                                                                  anaphora_sentence))

    mention_pair = mention_pair.as_array()

    # return feature vector without mention strings and without coreference label
    return mention_pair[2:13]



  # _extract_nps extracts NPs from an NLTK Tree
  # returns a list containing tuples of the form (sentence_id, start_of_np_span, end_of_np_span, position_of_head)
  def _extract_nps(self,tree,sentence_id) :
    nps = []
    # Stack for saving the subtrees that are yet to be processed
    subtrees = deque()

    subtrees.append(tree)

    # look for NPs as long as there is something in subtrees
    while subtrees :
      current_tree = subtrees.pop()

      try :
        if current_tree.label() == 'NP' or current_tree.label() == 'PRP$' :
          start_of_np_span = int(current_tree.leaves()[0])
          end_of_np_span = int(current_tree.leaves()[-1])
          position_of_head = self._extract_np_head(current_tree)
          nps.append((sentence_id,start_of_np_span,end_of_np_span,position_of_head))

      # If we are already at the leaves of the tree, tree.label() will raise an AttributeError
      except AttributeError :
        pass

      # adding subtrees of current_tree to stack
      try :
        for child_node in current_tree :
          if not isinstance(child_node,str) :
            # skipping appositives
            if child_node.label() == 'NP' and current_tree.label() == 'NP' :
              for grandchild_node in child_node :
                if not isinstance(grandchild_node,str) :
                  subtrees.appendleft(grandchild_node)
            else :
              subtrees.appendleft(child_node)

      # defective parse trees in the annotations are assigned None by the parser and will therefore
      # raise a TypeError
      except TypeError :
        pass

    return nps

  def _extract_mention_pairs_with_features(self,np_list,document) :
    # extract features that are independent of a second NP
    labeled_nps = []
    for np in np_list :
      np_sentence = document[np[0]]
      np_head = np[3]

      np_string = self._extract_string(np,np_sentence)
      np_is_pronoun = self._extract_if_pronoun(np,np_sentence)
      np_is_def = self._extract_if_def_np(np,np_sentence)
      np_is_dem = self._extract_if_dem_np(np,np_sentence)
      np_number = self._extract_number(np_head,np_sentence)
      np_sem_class = self._extract_sem_class(self._chose_synset(np_string))
      np_gender = self._extract_gender(np,np_sentence)
      np_ne_class = self._extract_ne_class(np,np_sentence)

      features = (np_string,
                  np_is_pronoun,
                  np_is_def,
                  np_is_dem,
                  np_number,
                  np_sem_class,
                  np_gender,
                  np_ne_class)

      labeled_nps.append(np + features)

      # NPs in labeled_nps are 12-tuples with the following content at index:
      #     0  - sentence ID
      #     1  - Start of NP Span
      #     2  - End of NP Span
      #     3  - Position of Head of NP
      #     4  - NP String
      #     5  - True if NP is a pronoun, else False
      #     6  - True if NP is definite, else False
      #     7  - True if NP is demonstrative, else False
      #     8  - True if NP is plural, else False
      #     9  - semantic class string (None if no class found)
      #     10 - gender string
      #     11 - named entity class string (None if not a named entity)

    training_instances = []

    # extract mention-pair features
    for index, anaphora in enumerate(labeled_nps) :
      for antecedent in reversed(labeled_nps[:index]) :
        sentence_anaphora = document[anaphora[0]]
        sentence_antecedent = document[antecedent[0]]

        # Mention-Pairs are extracted as explained in Soon et al.'s paper
        mention_pair = MentionPair.MentionPair(self._extract_features(antecedent,anaphora,sentence_antecedent,sentence_anaphora))
        training_instances.append(mention_pair.as_array())

        if mention_pair.is_coreferent() :
          break

    return training_instances

  def _extract_independent_features(self,np,np_sentence) :
    np_string = self._extract_string(np,np_sentence)
    np_is_pronoun = self._extract_if_pronoun(np,np_sentence)
    np_is_def = self._extract_if_def_np(np,np_sentence)
    np_is_dem = self._extract_if_dem_np(np,np_sentence)
    np_number = self._extract_number(np[3],np_sentence)
    np_sem_class = self._extract_sem_class(self._chose_synset(np_string))
    np_gender = self._extract_gender(np,np_sentence)
    np_ne_class = self._extract_ne_class(np,np_sentence)

    features = (np_string,
                np_is_pronoun,
                np_is_def,
                np_is_dem,
                np_number,
                np_sem_class,
                np_gender,
                np_ne_class)

      # RETURN: 12-tuple with the following content at index:
      #     0  - sentence ID
      #     1  - Start of NP Span
      #     2  - End of NP Span
      #     3  - Position of Head of NP
      #     4  - NP String
      #     5  - True if NP is a pronoun, else False
      #     6  - True if NP is definite, else False
      #     7  - True if NP is demonstrative, else False
      #     8  - True if NP is plural, else False
      #     9  - semantic class string (None if no class found)
      #     10 - gender string
      #     11 - named entity class string (None if not a named entity)

    return (np + features)


  # extracts features of two NPs and returns a tuple containing these features
  def _extract_features(self,np_i,np_j,sentence_i,sentence_j) :
    
    return (np_i[4],  # String of NP_i
            np_j[4],  # String of NP_j
            self._extract_distance(np_i,np_j),
            np_i[5],  # is NP_i pronoun
            np_j[5],  # is NP_j pronoun
            self._extract_if_string_match(np_i,np_j,sentence_i,sentence_j),
            np_j[6],   # is NP_j definite
            np_j[7],  # is NP_j demonstrative
            self._extract_if_number_agreement(np_i,np_j,sentence_i,sentence_j),
            self._extract_if_sem_class_agreement(np_i,np_j,sentence_i,sentence_j),
            self._extract_if_gender_agreement(np_i,np_j,sentence_i,sentence_j),
            self._extract_if_proper_names(np_i,np_j,sentence_i,sentence_j),
            self._extract_if_alias(np_i,np_j,sentence_i,sentence_j),
            self._extract_if_coreferent(np_i,np_j,sentence_i,sentence_j))

  def _extract_features_for_prediction(self,antecedent,anaphora,sentence_antecedent,sentence_anaphora) :

    return (np_i[4],  # String of antecedent
            np_j[4],  # String of anaphora
            self._extract_distance(np_i,np_j),
            np_i[5],  # is antecedent pronoun
            np_j[5],  # is anaphora pronoun
            self._extract_if_string_match(np_i,np_j,sentence_i,sentence_j),
            np_j[6],   # is anaphora definite
            np_j[7],  # is anaphora demonstrative
            self._extract_if_number_agreement(np_i,np_j,sentence_i,sentence_j),
            self._extract_if_sem_class_agreement(np_i,np_j,sentence_i,sentence_j),
            self._extract_if_gender_agreement(np_i,np_j,sentence_i,sentence_j),
            self._extract_if_proper_names(np_i,np_j,sentence_i,sentence_j),
            self._extract_if_alias(np_i,np_j,sentence_i,sentence_j),
            None)

  # returns the string of an NP generated by _extract_np() 
  def _extract_string(self,np,sentence) :
    return " ".join(sentence.words[np[1]:np[2]+1])

  # return the distance (in sentences) of two NPs generated by _extract_np()
  def _extract_distance(self,np_i,np_j) :
    return int(np_j[0] - np_i[0])

  # returns if an NP extracted by _extract_np() is a pronoun
  def _extract_if_pronoun(self,np,sentence) :
    # pronouns consist (usually) of one word, so we check if start of NP = end of NP
    if np[1] == np[2] :
      pronoun_tags = {'PRP', 'PRP$', 'WP', 'WP$'}
      pos_tag_of_np = sentence.pos_tags[np[1]]
      # return if NP is a pronoun (i.e. if it is tagged as a pronoun)
      return pos_tag_of_np in pronoun_tags

    # NP is not a pronoun (because it consits of more than one word)
    return False

  # removes articles (a, an, the) and demonstrative pronouns (this, these) of two NPs extracted by
  # _extract_np() and returns then if the strings match
  def _extract_if_string_match(self,np_i,np_j,sentence_np_i,sentence_np_j) :
    articles = {'a','an','the','this','these','that','those'}
    string_np_i = [sentence_np_i.words[k] for k in range(np_i[1],np_i[2]+1)
                                          if sentence_np_i.words[k].lower() not in articles]
    string_np_j = [sentence_np_j.words[k] for k in range(np_j[1],np_j[2]+1)
                                          if sentence_np_j.words[k].lower() not in articles]

    return string_np_i == sentence_np_j

  # returns if an NP generated by _extract_np() is a definite NP (i.e. starts with 'the')
  def _extract_if_def_np(self,np,sentence) :
    first_word = sentence.words[np[1]]
    return first_word.lower() == 'the'

  # returns if an NP generated by _extract_np() is a demonstrative NP 
  # (i.e. starts with 'this', 'these', 'that' or 'those')
  def _extract_if_dem_np(self,np,sentence) :
    first_word = sentence.words[np[1]]
    demonstratives = {'this','that','theese','those'}
    return first_word.lower() in demonstratives

  # returns if two NPs extracted by _extract_np() agree in number
  def _extract_if_number_agreement(self,np_i,np_j,sentence_np_i,sentence_np_j) :
    # extract head of NP
    np_i_head = np_i[3]
    np_j_head = np_j[3]

    # if both heads are not found, we assume that gender features agree
    if (np_i_head == None) and (np_j_head == None) :
      return True
    # if one head is not found, we assume that gender features do not agree
    if (np_i_head == None) or (np_j_head == None) :
      return False

    # extract number of head of NP
    np_i_number = np_i[8]
    np_j_number = np_j[8]

    return np_i_number == np_j_number


  # returns sentence position of the head of an NP generated by _extract_np()
  def _extract_np_head(self,nptree) :
    # if NP consists of only one terminal, this terminal must be the head
    if len(nptree.leaves()) == 1 :
      return int(nptree.leaves()[0])

    # if NP consists of more than one word, the head is the rightmost nominal terminal of the 
    # rightmost sub-NP
    else :
      np_terminals = nptree.leaves()
      nominal_terminals = {'NN','NNS','NNP','NNPS'}

      while True :
        rightmost_np_found = False
        for subtree in reversed(nptree) :
          # NP-Head found
          if subtree.label() in nominal_terminals :
            return int(subtree.leaves()[0])
          # rightmost NP-subtree is next tree to look for nominal terminal
          if (subtree.label() == 'NP') and (not rightmost_np_found) :
            nptree = subtree
            rightmost_np_found = True

        # NP-head not detectable
        if 'NP' not in [x.label() for x in nptree] :
          return None

  # extracts the number of a head of an NP generated by _extract_np_head()
  # returns False if NP is singular and True if NP is plural
  def _extract_number(self,head,sentence) :
    # check if head is found
    if head is None :
      return
    # if head is a pronoun, we look in a list of plural pronouns if it is plural
    if sentence.pos_tags[head] in {'PRP','PRP$'} :
      plural_pronouns = {'we','us','our','ours','ourself','ourselves','they','them','their',
                         'theirs','theirself','theirselves'}
      # return if pronoun is is plural_pronouns
      return (sentence.words[head].lower() in plural_pronouns)

    # if it is a common or proper noun, we check if the plural version of POS-Tag is used
    plural_pos_tags = {'NNS','NNPS'}
    return (sentence.pos_tags[head] in plural_pos_tags)


  # returns True if two NPs extracted by _extract_np() share a semantic class or, if both classes are
  # unknown, returns True if the strings match
  # returns False if the semantic classes differ
  # returns unknown if both classes are unknown and the strings do not match
  def _extract_if_sem_class_agreement(self,np_i,np_j,sentence_np_i,sentence_np_j) :
    np_i_sem_class = np_i[9]
    np_j_sem_class = np_j[9]

    # both classes are unknown
    if (np_i_sem_class is None) and (np_j_sem_class is None) :
      # check if strings match
      if np_i[4] == np_j[4] :
        return True
      # strings do not match
      return 'unknown'

    elif (np_i_sem_class is None) or (np_j_sem_class is None) :
      return False

    # check for agreement
    person_hyponyms = {'person','male','female'}
    object_hyponyms = {'object','organization','location','date','time','money','percent'}

    # classes agree
    if (np_i_sem_class in person_hyponyms and np_j_sem_class in person_hyponyms) or \
       (np_i_sem_class in object_hyponyms and np_j_sem_class in object_hyponyms) :
       return True

    # classes do not agree
    return False


  # _chose_synset() extracts the first (= most frequent) nominal synset for an NP-head
  def _chose_synset(self,np_string) :
    for synset in wordnet.synsets(np_string) :
      if synset.name().split('.')[1] == 'n':
        return synset

    return None


  # _extract_sem_class() looks if a noun (given in form of A WordNet Synset) contains to one of the
  # following classes and returns the class:
  # 'person': 'male', 'female'
  # 'object': 'organization', 'location', 'date', 'time', 'money', 'percent'
  def _extract_sem_class(self,synset) :
    # check if synset is found
    if synset is None :
      return None

    extract_hypernyms = lambda s: s.hypernyms()
    semantic_classes = {'person','male','female','object','organization','location','date','time'
                        'money','percent'}

    # extract all hypernyms
    all_hypernyms = [hyper.name() for hyper in list(synset.closure(extract_hypernyms))]

    # return first class found (= most specific class)
    for hypernym in all_hypernyms :
      if hypernym.split('.')[0] in semantic_classes :
        return hypernym.split('.')[0]

    # class not found
    return None


  # returns True if both NPs agree in gender, else false
  # returns unknown if gender of both NPs is unknown
  def  _extract_if_gender_agreement(self,np_i,np_j,sentence_np_i,sentence_np_j) :
    gender_np_i = np_i[10]
    gender_np_j = np_j[10]
      
    # if at least one gender feature is unknown, gender agreement is unknown
    if (gender_np_i == 'unknown') or (gender_np_j == 'unknown') :
      return 'unknown'
      
    # test if gender features are equal
    return gender_np_i == gender_np_j


  # extracts the gender of an NP 
  def _extract_gender(self,np,sentence) :
    # first we try to use some heuristics to extract gender feature
    fem_designators = {'she','mrs.','miss','ms.','madam','lady'}
    masc_designators = {'he','mr.','sir'}
    np_head = np[3]
    # if head is not found, we check if one of the words in the NP is one of the designators
    if np_head == None :
      for word in sentence.words[np[1]:np[2]] :
        if word.lower() in fem_designators :
          return 'feminine'
        if word.lower() in masc_designators :
          return 'masculine'

      # look if whole NP is in gender frequencies
      np_string = ' '.join(sentence.words[np[1]:np[2]]).lower()
      try :
        frequencies = self.gender_frequencies[np_string]
      except KeyError :
        return 'unknown'

    
    else :
      # look if one of the designators is before head noun
      for word in sentence.words[np[1]:np_head+1] :
        if word.lower() in fem_designators :
          return 'feminine'
        if word.lower() in masc_designators :
          return 'masculine'
        
      # look up with which gender the NP-head is most often associated
      try :
        frequencies = self.gender_frequencies[sentence.words[np_head]]
      except KeyError :
        return 'unknown'
    
    max_freq = -1
    max_i = -1
    for i,freq in enumerate(frequencies) :
      if freq > max_freq :
        max_freq = freq
        max_i = i
    
    # index 0 is masculine
    if max_i == 0:
      return 'masculine'
    # index 1 is feminine
    if max_i == 1:
      return 'feminine'
    # index 2 is neuter
    return 'neuter'

  # returns True if both NPs are proper names, else False
  # to check if an NP is a proper name, the POS-Tag of the head noun is checked
  def _extract_if_proper_names(self,np_i,np_j,sentence_np_i,sentence_np_j) :
    np_i_head = np_i[3]
    np_j_head = np_j[3]
    proper_noun_tags = {'NNP','NNPS'}

    if (np_i_head == None) or (np_j_head == None) :
      return False
    
    # both NPs are proper nouns
    if (sentence_np_i.pos_tags[np_i_head] in proper_noun_tags) and \
       (sentence_np_j.pos_tags[np_j_head] in proper_noun_tags) :
      return True
      
    # at least one NP is not a proper noun
    return False


  # returns True if one NP is an alias of the other NP
  def _extract_if_alias(self,np_i,np_j,sentence_np_i,sentence_np_j) :
    # extract named entity class for both NPs
    np_i_ne_class = np_i[11]
    np_j_ne_class = np_j[11]

    # if at least one NP is not a named entity, NPs are not alias of each other
    if (not np_i_ne_class) or (not np_j_ne_class) :
      return False

    # if NPs are from differtent classes, they cannot be alias of each other
    if np_i_ne_class != np_j_ne_class :
      return False

    # check if persons are alias
    if np_i_ne_class == 'PERSON' :
      return sentence_np_i.words[np_i[2]] == sentence_np_j.words[np_j[2]]

    # check if organizations are alias
    if np_i_ne_class == 'ORG' :
      # determine whcih string is the acronym and which one is the full name
      len_np_i = 0
      len_np_j = 0
      for word in sentence_np_i.words[np_i[1]:np_i[2]+1] :
        len_np_i += len(word)
      for word in sentence_np_j.words[np_j[1]:np_j[2]+1] :
        len_np_j += len(word)

      # NP_i is the full name and NP_j is the acronym
      if len_np_i > len_np_j :
        return self._check_if_acronym(np_i,np_j,sentence_np_i,sentence_np_j)

      # NP_j is the full_name and NP_i is the acronym
      else :
        return self._check_if_acronym(np_j,np_i,sentence_np_j,sentence_np_i)

    return False


  # returns the named entity class of an NP (if NP is a named entity)
  def _extract_ne_class(self,np,sentence) :
    ne_class = None
    for annotation in sentence.named_entities[np[1]:np[2]+1] :
      # seperate class from position
      position_class = annotation.split('-')
      # if len of position_class is 1, NP is not a named entity
      if len(position_class) == 1 :
        return False

      # set class of named entity
      if ne_class is None :
        ne_class = position_class[1]

      # more than one named entity in NP, so NP is not a single named entity
      if ne_class != position_class[1] :
        return False

    return ne_class

  # checks for two named entities of type 'ORG' if one is the acronym of the other
  def _check_if_acronym(self,full_name,acronym,sentence_full_name,sentence_acronym) :
    generated_acronym = ''
    generated_dotted_acronym = ''
    generated_upper_acronym = ''
    generated_upper_dotted_acronym = ''
    # generate acronym without last word (last word could be an postmodifier like 'corp.')
    for word in sentence_full_name.words[full_name[1]:full_name[2]] :
      generated_acronym += word[0].lower()
      generated_dotted_acronym += '{}.'.format(word[0].lower())
      if word[0].isupper() :
        generated_upper_acronym += word[0].lower()
        generated_upper_dotted_acronym += '{}.'.format(word[0].lower())

    acronym = ''.join(sentence_acronym.words[acronym[1]:acronym[2]+1]).lower()
    if acronym == generated_acronym or \
       acronym == generated_dotted_acronym or \
       acronym == generated_upper_acronym or \
       acronym == generated_upper_dotted_acronym :
      return True

    # add last word
    last_word = sentence_full_name.words[full_name[2]]
    generated_acronym += last_word[0].lower()
    generated_dotted_acronym += '{}.'.format(last_word[0].lower())
    if last_word.isupper() :
      generated_upper_acronym += last_word[0].lower()
      generated_upper_dotted_acronym += '{}.'.format(last_word[0].lower())

    if acronym == generated_acronym or \
       acronym == generated_dotted_acronym or \
       acronym == generated_upper_acronym or \
       acronym == generated_upper_dotted_acronym :
      return True

    return False

  
  # returns True if two NPs are coreferent according to OntoNotes Annotation
  def _extract_if_coreferent(self,np_i,np_j,sentence_np_i,sentence_np_j) :
    np_i_entity_id = -1
    np_j_entity_id = -1
    # determine entity ID for NP_i
    for mention in sentence_np_i.coref_spans :
      if np_i[1:3] in mention :
        np_i_entity_id = mention[0]
        break

    if np_i_entity_id == -1 :
      return False

    # determine entity ID for NP_j
    for mention in sentence_np_j.coref_spans :
      if np_j[1:3] in mention :
        np_j_entity_id = mention[0]
        break

    if np_j_entity_id == -1 :
      return False

    return np_i_entity_id == np_j_entity_id

if __name__ == '__main__':

  # ERROR: not right amount of arguments passed
  if len(sys.argv) < 3 :
    print("Synopsis: FeatureExtractor.py INPUTFILE/FOLDER OUTPUTFILE", file=sys.stderr)
    print("INPUTFILE is a OntoNote-File / FOLDER is a folder containing OntoNote-Files", file=sys.stderr)
    print("OUTPUTFILE is the file, where the feature matrix should be saved", file=sys.stderr)
    sys.exit(1)

  else :
    pathname = sys.argv[1]
    feature_extractor = FeatureExtractor(pathname)
    iterator = feature_extractor.training_instances_iterator()
    for instance in next(iterator) :
      print(instance.as_tab_seperated_line())

