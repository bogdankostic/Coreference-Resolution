# Coreference Resolution
# Final Project
# Bogdan Kostić, 785726, 19 March 2019
# Python 3.7.2

# Objects of the MentionPair class represent the features for an extracted mention-pair.

class MentionPair:
    
  def __init__(self, feature_tuple) :

    self.mention_i: str = feature_tuple[0]
    self.mention_j: str = feature_tuple[1]
    self.distance: int = feature_tuple[2]
    self.i_pronoun: bool = feature_tuple[3]
    self.j_pronoun: bool = feature_tuple[4]
    self.string_match: bool = feature_tuple[5]
    self.def_np_j: bool = feature_tuple[6]
    self.dem_np_j: bool = feature_tuple[7]
    self.number_agreement: bool = feature_tuple[8]
    self.semantic_class_agreement = feature_tuple[9]
    self.gender_agreement = feature_tuple[10]
    self.both_proper_names: bool = feature_tuple[11]
    self.alias: bool = feature_tuple[12]
    self.coreferent = feature_tuple[13]

  def is_coreferent(self) :
    return self.coreferent

  def as_tab_seperated_line(self) :
    return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"\
            .format(self.mention_i,
                    self.mention_j,
                    self.distance,
                    '+' if self.i_pronoun else '-',
                    '+' if self.j_pronoun else '-',
                    '+' if self.string_match else '-',
                    '+' if self.def_np_j else '-',
                    '+' if self.dem_np_j else '-',
                    '+' if self.number_agreement else '-',
                    '+' if self.semantic_class_agreement == True else '-' if self.semantic_class_agreement == False else 'unknown',
                    '+' if self.gender_agreement == True else '-' if self.gender_agreement == False else 'unknown',
                    '+' if self.both_proper_names else '-',
                    '+' if self.alias else '-',
                    '+' if self.coreferent else '-')

  def as_array(self) :
    return [self.mention_i,
            self.mention_j,
            self.distance,
            '+' if self.i_pronoun else '-',
            '+' if self.j_pronoun else '-',
            '+' if self.string_match else '-',
            '+' if self.def_np_j else '-',
            '+' if self.dem_np_j else '-',
            '+' if self.number_agreement else '-',
            '+' if self.semantic_class_agreement == True else '-' if self.semantic_class_agreement == False else 'unknown',
            '+' if self.gender_agreement == True else '-' if self.gender_agreement == False else 'unknown',
            '+' if self.both_proper_names else '-',
            '+' if self.alias else '-',
            '+' if self.coreferent else '-']





