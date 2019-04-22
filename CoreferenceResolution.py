# -*- coding: utf-8 -*-
# Coreference Resolution
# Final Project
# Bogdan Kostić, 785726, 19 March 2019
# Python 3.7.2

import sys 

import FeatureExtractor
import CoreferenceClassifier

# The CoreferenceResolution class iterates over OntoNotes files provided by FeatureExtractor and
# outputs coreferene predictions for each file.
class CoreferenceResolution:

  def __init__(self,feature_extractor,classifier,output_pred_filename) :
    conll_document_iterator = feature_extractor.document_iterator()
    gold_file = open('CONLL_GOLD_KEY','w')
    pred_file = open(output_pred_filename,'w')

    for document, mentions in conll_document_iterator :
      if document :
        pred_file.write("#begin document ({}); part {}\n".format(document[0].document_id,
                                                                 str(document[0].part_number).zfill(3)))

        gold_file.write("#begin document ({}); part {}\n".format(document[0].document_id,
                                                                 str(document[0].part_number).zfill(3)))

        # create key for scorer
        self._write_key_file(document,gold_file)

        mention_entity_mapping = {}
        current_entity_index = 0
        # closest-first clustering is used, i.e. the nearest possible antecedent of an anaphor that 
        # is classified as coreferent to the anaphora will be the antecedent
        for index, anaphor in enumerate(mentions) :
          for antecedent in reversed(mentions[:index]) :
            mention_pair_feature_vector = feature_extractor.extract_feature_vector(antecedent,anaphor,document)
            # make prediction
            pred = classifier.predict_mention_pair(mention_pair_feature_vector)
            # both mentions are coreferent
            if pred == '+' :

              # a new coreference chain is initialized, if antecedent is not in the mapping and 
              # therefore does not possess of an entity id
              if antecedent not in mention_entity_mapping :
                mention_entity_mapping[antecedent] = current_entity_index
                current_entity_index += 1

              # assign entity id to anaphor based on antecedent
              mention_entity_mapping[anaphor] = mention_entity_mapping[antecedent]

        self._write_pred_file(document,mention_entity_mapping,pred_file)

        pred_file.write("#end document\n")
        gold_file.write("#end document\n")

    pred_file.close()
    gold_file.close()

  # auxiliary function to write gold key file
  def _write_key_file(self,document,file) :
    for sentence in document :
      self._write_sentence_to_file(sentence,sentence.coref_spans,file)

  # auxiliary function to write prediction file
  def _write_pred_file(self,document,mention_entity_mapping,file) :
    coref_spans = [set() for _ in document]
    for mention, entity_id in mention_entity_mapping.items() :
      sentence_id = mention[0]
      coref_span = (entity_id,(mention[1],mention[2]))
      coref_spans[sentence_id].add(coref_span)

    for index, sentence in enumerate(document) :
      self._write_sentence_to_file(sentence,coref_spans[index],file)

  # auxiliary function that writes coreference chains in correct format
  def _write_sentence_to_file(self,sentence,coref_spans,file) :

    rows_to_write = [[sentence.document_id] for _ in sentence.words]
    for coref_span in coref_spans :
      # START OF MENTION SPAN
      # no mention starts at this position
      if len(rows_to_write[coref_span[1][0]]) == 1 :
        # span length is only one
        if coref_span[1][0] == coref_span[1][1] :
          rows_to_write[coref_span[1][0]].append("({})".format(coref_span[0]))
          continue
        # span length is greater than one
        rows_to_write[coref_span[1][0]].append("({}".format(coref_span[0]))

      # another mention already starts at this position
      else :
        # span length is only one
        if coref_span[1][0] == coref_span[1][1] :
          rows_to_write[coref_span[1][0]][1] += "|({})".format(coref_span[0])
          continue
        # span length is greater than one
        rows_to_write[coref_span[1][0]][1] += "|({}".format(coref_span[0])

      # END OF MENTION SPAN
      # no mention ends at this position
      if len(rows_to_write[coref_span[1][1]]) == 1 :
        rows_to_write[coref_span[1][1]].append("{})".format(coref_span[0]))
      # another mention already ends at this position :
      else :
        rows_to_write[coref_span[1][1]][1] += "|{})".format(coref_span[0])

    for row in rows_to_write :
      if len(row) == 1 :
        file.write("{} -\n".format(row[0]))
      else :
        file.write("{} {}\n".format(row[0],row[1]))
    file.write("\n")


if __name__ == '__main__' :

  # ERROR: not right amount of arguments passed
  if len(sys.argv) < 3 :
    print("Synopsis: FeatureExtractor.py INPUTFILE/FOLDER OUTPUTFILE (CLASSIFIER_BINARY)", file=sys.stderr)
    print("INPUTFILE is an OntoNote-File / FOLDER is a folder containing OntoNote-Files", file=sys.stderr)
    print("OUTPUTFILE is an OntoNote-File", file=sys.stderr)
    print("(CLASSIFIER_BINARY) is a binary containing an already trained model", file=sys.stderr)
    print("The dault classifier is SVM.", file=sys.stderr)
    sys.exit(1)

  else :
    path_to_test_instances = sys.argv[1]
    output_filename = sys.argv[2]

    if len(sys.argv) == 4 :
      path_to_model = sys.argv[3]
    else :
      path_to_model = 'Trained_Models/SVM'

    test_instances = FeatureExtractor.FeatureExtractor(path_to_test_instances)
    classifier = CoreferenceClassifier.CoreferenceClassifier.load_binary(path_to_model)

    CoreferenceResolution(test_instances,classifier,output_filename)
