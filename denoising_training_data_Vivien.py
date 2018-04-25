import os
import numpy as np
from spacy import util
from snorkel import SnorkelSession
from snorkel.parser.spacy_parser import Spacy
from snorkel.parser import TSVDocPreprocessor, CorpusParser
from snorkel.models import candidate_subclass
from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.matchers import *
from snorkel.models import Document, Sentence
from snorkel.models import candidate_subclass
from snorkel.annotations import LabelAnnotator
from collections import defaultdict
from snorkel.models import candidate_subclass
import re
from commonregex import email
from commonregex import street_address
from find_job_titles import FinderAcora
from snorkel.lf_helpers import (
    get_left_tokens, get_right_tokens, get_between_tokens,
    get_text_between, get_tagged_text,
)

from snorkel.lf_helpers import (
    get_tagged_text,
    rule_regex_search_tagged_text,
    rule_regex_search_btw_AB,
    rule_regex_search_btw_BA,
    rule_regex_search_before_A,
    rule_regex_search_before_B,
)


# A ContextSpace defines the "space" of all candidates we even potentially consider; in this case we use the Ngrams subclass, and look for all n-grams up to 7 words long

session = SnorkelSession()

doc_preprocessor = TSVDocPreprocessor('/Users/fanglinchen/Desktop/PersonalDataStack/DeepScrub/DeepScrub/algorithms/input.tsv', max_docs=350) 
corpus_parser = CorpusParser(parser=Spacy())
corpus_parser.apply(doc_preprocessor)

Sensitive = candidate_subclass('Sensitive', ['sensitive'], values = ['person', 'job', 'event', 
                                                                    'place', 'date', 'time', 
                                                                    'product', 'email', 'phone', 
                                                                    'quantity', 'address', 'url', 
                                                                    'org', 'file', 'password', False])
# generating candidates. 
ngrams = Ngrams(n_max=6)
ngramMatcher = NgramMatcher(longest_match_only = False)


cand_extractor = CandidateExtractor(
    Sensitive, 
    [ngrams],
    [ngramMatcher],
    symmetric_relations=False
)
sents = session.query(Sentence).all()
cand_extractor.apply(sents, split=0)
train_cands = session.query(Sensitive).filter(Sensitive.split == 0).all()
finder = FinderAcora()

def find(array, word):
    return [i for i, each in enumerate(array) if each == word]

# ner
def LF_product(c):
    if len(c.sensitive.get_attrib_tokens("words")) == len(find(c.sensitive.get_attrib_tokens("ner_tags"), "PRODUCT")):
        print "PRODUCT:"+ c.sensitive.get_span()
        return "product"

# regex
def LF_job(c):
    mgroups = finder.findall(c.sensitive.get_span())
    if len(mgroups) > 0:
        print "job:"+ str(mgroups[0])
        return "job"

# left_tokens
def LF_file_before(c):

    if "see attached file :" == " ".join(list(get_left_tokens(c[0], window=4))):
        print c.sensitive.get_span()
        return "file"

# regex
def LF_file(c):
    file_regex = re.compile("^[\w,\s-]+\.[A-Za-z]{3}$")
    mgroups = file_regex.findall(c.sensitive.get_span())
    
    if len(mgroups) > 0:
        print "file:"+ str(mgroups[0])
        return "file"

# regex       
def LF_phone(c):
    phone_regex = re.compile("(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})")
    to_remove = find(c.sensitive.get_attrib_tokens("ner_tags"), "O")
   
    cleaned_number = ''
    for i, word in enumerate(c.sensitive.get_attrib_tokens("words")):
        if i not in to_remove:
            if c.sensitive.get_attrib_tokens("ner_tags")[i] == "CARDINAL":
                cleaned_number += word
            
    if cleaned_number != '':
        try:
            matched_phone = phone_regex.findall(cleaned_number)[0]
            
            if matched_phone == cleaned_number:
                
                print "phone:" + c.sensitive.get_span()
                return "phone"
        except IndexError:
            pass

# regex 
def LF_email(c):

    mgroups = email.findall(c.sensitive.get_span())
    
    if len(mgroups) > 0:
        print "email:"+ str(mgroups[0])
        return "email"

# get_right_token
# get location or people by @
def LF_location_people_before(c):
    if "@" == " ".join(list(get_right_tokens(c[0], window=1))):
        print c.sensitive.get_span()
        return "whoever/wherever"


# regex
def LF_address(c):
    
    mgroups = street_address.findall(c.sensitive.get_span())
    
    if len(mgroups) > 0:
        print "address:"+ str(mgroups[0])
        return "address"

# ner                  
def LF_org(c):
    if len(c.sensitive.get_attrib_tokens("words")) == len(find(c.sensitive.get_attrib_tokens("ner_tags"), "ORG")):
        print "ORG:"+ c.sensitive.get_span()
        return 'org'

# ner       
def LF_event(c):
    if len(c.sensitive.get_attrib_tokens("words")) == len(find(c.sensitive.get_attrib_tokens("ner_tags"), "EVENT")):
        print "EVENT:"+ c.sensitive.get_span()
        return 'event'

# not sure left or right
# get special event: lunch and dinner
def LF_event_lunch_dinner(c):
    if "lunch" or "dinner"==" ".join(list(get_left_tokens(c[0],window=1))):
        print "EVENT:"c.sensitive.get_span()
        return 'event'


# regex              
def LF_url(c):
    url_regex = re.compile(ur'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
    
    mgroups = url_regex.findall(c.sensitive.get_span())
    
    if len(mgroups) > 0:
        print "url:"+ str(mgroups[0])
        return "url"

# ner               
def LF_time(c):
    if len(c.sensitive.get_attrib_tokens("words")) == len(find(c.sensitive.get_attrib_tokens("ner_tags"), "TIME")):
        print "TIME:"+ c.sensitive.get_span()
        return 'time'

# am/pm
def LF_time_pm(c):
    if "pm "==" ".join(list(get_right_tokens(c[0],window=1))):
        print "TIME"
        return 'time'
        
def LF_time_am(c):
    if "am " not " am "==" ".join(list(get_right_tokens(c[0],window=1))):
        print "TIME"
        return 'time'

# ner
def LF_location(c):
    if len(c.sensitive.get_attrib_tokens("words")) == len(find(c.sensitive.get_attrib_tokens("ner_tags"), "LOC")):
        print "place:"+ c.sensitive.get_span()
        print c.sensitive.get_attrib_tokens("ner_tags")
        return 'place'

# ner    
def LF_date(c):
    if len(c.sensitive.get_attrib_tokens("words")) == len(find(c.sensitive.get_attrib_tokens("ner_tags"), "DATE")):
        print "DATE:"+ c.sensitive.get_span()
        print c.sensitive.get_attrib_tokens("ner_tags")
        return 'date'

# today/tomorrow
def LF_date_today_tomorrow(c):
    if "today" or "tomorrow"==" ".join(list(get_left_tokens(c[0],window=1))):
        print "DATE:" + c.sensitive.get_span()
        return 'date'

# whether need to collect Mon-Sunday??

# ner  
def LF_person(c):
    if len(c.sensitive.get_attrib_tokens("words")) == len(find(c.sensitive.get_attrib_tokens("ner_tags"), "PERSON")):
        print "PERSON:"+ c.sensitive.get_span()
        print c.sensitive.get_attrib_tokens("ner_tags")
        return 'person'
    
LFs = [LF_time, LF_date, LF_location, LF_person, LF_org, LF_url, LF_phone, LF_product, LF_event, LF_email, LF_address, LF_job, LF_file, LF_file_before] 

labeler = LabelAnnotator(lfs=LFs)
np.random.seed(1701)
L_train = labeler.apply(split=0)
print L_train.lf_stats(session, )

L_train.todense()

from snorkel.learning import GenerativeModel

gen_model = GenerativeModel()
gen_model.train(L_train, cardinality=3)

train_marginals = gen_model.marginals(L_train)

# assert np.all(train_marginals.sum(axis=1) - np.ones(3) < 1e-10)
# train_marginals


from snorkel.annotations import save_marginals, load_marginals

save_marginals(session, L_train, train_marginals)

from snorkel.annotations import FeatureAnnotator
featurizer = FeatureAnnotator()

F_train = featurizer.apply(split=0)
from snorkel.learning import SparseLogisticRegression
disc_model_sparse = SparseLogisticRegression(cardinality=Spouse.cardinality)

disc_model_sparse.train(F_train, train_marginals, n_epochs=100, print_freq=10, lr=0.001)


train_labels = [1, 2, 1]
correct, incorrect = disc_model_sparse.error_analysis(session, F_train, train_labels)
print "Accuracy:", disc_model_sparse.score(F_train, train_labels)

test_marginals = disc_model_sparse.marginals(F_train)
print test_marginals
