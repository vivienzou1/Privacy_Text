# ./run.sh
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

# regex
def  LF_job_1(c):
    poli_regex = re.compile("republician"|"politican"|"doctor"|"engineer "|"student"|"professor "|"lawyer"|"scientist"|"cashier"|
                            " worker"|"nurse"|"waiter"|"waitress"|"assistant"|"manager"|"mover"|"housekeeper"|"teacher"|"chief"|
                            "soldier"|"clerk"|"businessman"|"ceo"|"cfo")
    mgroups = poli_regex.findall(c.sensitive.get_span())
    
    if len(mgroups) > 0:
        print "file:"+ str(mgroups[0])
        return "file"

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

# get_left_token
# get location or people by @
# what to return??
def LF_address_people_before(c):
    if "@" == " ".join(list(get_left_tokens(c[0], window=1))):
        print c.sensitive.get_span()
        return "whoever/wherever"


# regex
def LF_address(c):
    
    mgroups = street_address.findall(c.sensitive.get_span())
    
    if len(mgroups) > 0:
        print "address:"+ str(mgroups[0])
        return "address"

# get_right_token
# address with ave/blvd
def LF_address_ave_blvd(c):
    if " ave " or "blvd"==" ".join(list(get_right_tokens(c[0], window=1))):
        print "ADDRESS:"+ c.sensitive.get_span()
        return "address"

# ner                  
def LF_org(c):
    if len(c.sensitive.get_attrib_tokens("words")) == len(find(c.sensitive.get_attrib_tokens("ner_tags"), "ORG")):
        print "ORG:"+ c.sensitive.get_span()
        return 'org'

# regex
# org: school
def LF_org_school(c):
    school_regex = re.compile("school"|"schools"|"college"|"colleges"|"university"|"universities")
    mgroups = school_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "ORG:"+ c.sensitive.get_span()
        return 'org'

# regex
# org: companies
def LF_org_company(c):
    company_regex = re.compile("company"|"companies"|"institute"|"partnership"|"institution "|"corporation"|"corporations"|"co-op")
    mgroups = company_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "ORG:"+ c.sensitive.get_span()
        return 'org'

# regex
# org: govern
def LF_org_govern(c):
    govern_regex = re.compile("government"|"dictatorship"|"monarchy"|"democracy"|"communism"|"republic")
    mgroups = govern_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "ORG:"+ c.sensitive.get_span()
        return 'org'

# get_right_token
# org: police/fire... department
def LF_org_dept(c):
    if "department"==" ".join(list(get_right_tokens(c[0], window=1))):
        print "ORG:"+ c.sensitive.get_span()
        return "org"

# ner       
def LF_event(c):
    if len(c.sensitive.get_attrib_tokens("words")) == len(find(c.sensitive.get_attrib_tokens("ner_tags"), "EVENT")):
        print "EVENT:"+ c.sensitive.get_span()
        return 'event'

# regex
# get special event: lunch and dinner
def LF_event_lunch_dinner(c):
    lunch_dinner_regex = re.compile("lunch"|"dinner")
    mgroups = lunch_dinner_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "EVENT:"+ c.sensitive.get_span()
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

# get_right_token
# am/pm
def LF_time_pm(c):
    if "pm "==" ".join(list(get_right_tokens(c[0],window=1))):
        print "TIME" + c.sensitive.get_span()
        return 'time'
        
def LF_time_am(c):
    if "am " not " am "==" ".join(list(get_right_tokens(c[0],window=1))):
        print "TIME" + c.sensitive.get_span()
        return 'time'

# ner
def LF_location(c):
    if len(c.sensitive.get_attrib_tokens("words")) == len(find(c.sensitive.get_attrib_tokens("ner_tags"), "LOC")):
        print "place:"+ c.sensitive.get_span()
        print c.sensitive.get_attrib_tokens("ner_tags")
        return 'place'

# regex
# specific location name, exclued: keyword in org
def LF_location_spe(c):
    location_1_regex = re.compile("hospital"|"supermarket"|" mall "|"home"|"bathroom"|"restaurant"|"hotel"|"airport"|"station"|
                                   "park "|"museum"|"club"|"parking lot"|"airbnb"|"gym"|"library"|"studio"|"theater"|"party"
                                   |"parties"|"beach"|"bar"|"class"|"gallery"|"fair"|"garden"|"concert")
    
    mgroups = location_1_regex.findall(c.sensitive.get_span())
    
    if len(mgroups) > 0:
        print "PLACE:"+ c.sensitive.get_span()
        return "place"

# get_right_token
# general location name
def LF_location_gen(c):
    if "salon" or "community" or "store" or "resort" or "shop"==" ".join(list(get_right_tokens(c[0],window=1))):
        print "place:" + c.sensitive.get_span()
        return 'place'

# ner    
def LF_date(c):
    if len(c.sensitive.get_attrib_tokens("words")) == len(find(c.sensitive.get_attrib_tokens("ner_tags"), "DATE")):
        print "DATE:"+ c.sensitive.get_span()
        print c.sensitive.get_attrib_tokens("ner_tags")
        return 'date'

# regex
# month with/out abbreviation
def LF_date_month_abb(c):
    date_month_abb_regex = re.compile("Jan."|"Feb."|"Mar."|"Apr."|"Aug."|"Sept."|"Oct."|"Nov."|"Dec."|"May"|"June"|"July"|
                          "January"|"February"|"March"|"April"|"August"|"September"|"October"|"November"|"December")
    mgroups = date_month_abb_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "DATE:" + c.sensitive.get_span()
        return 'date'

# regex
# today/tomorrow
def LF_date_today_tomorrow(c):
    today_tomorrow_regex = re.compile("today"|"tomorrow")
    mgroups = today_tomorrow_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "DATE:" + c.sensitive.get_span()
        return 'date'

# regex
# whether need to collect Mon-Sunday??
def LF_date_day(c):
    day_regex = re.compile("monday"|"tuesday"|"wednesday"|"thursday"|"friday"|"saturday"|"sunday")
    mgroups = day_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "DAY" + c.sensitive.get_span()
        return 'date'

# ner  
def LF_person(c):
    if len(c.sensitive.get_attrib_tokens("words")) == len(find(c.sensitive.get_attrib_tokens("ner_tags"), "PERSON")):
        print "PERSON:"+ c.sensitive.get_span()
        print c.sensitive.get_attrib_tokens("ner_tags")
        return 'person'

# regex
# family memeber
def LF_person_family(c):
    family_regex = re.compile("mother"|"father"|"sister"|"brother"|" son "|"daughter"|"uncle "|"anut"|"cousin"|"grandmother"|"grandfather")
    mgroups = family_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "PERSON:" + c.sensitive.get_span()
        return 'person'

# regex
# gender
def LF_person_gender(c):
    gender_regex = re.compile("femail"|"male")
    mgroups = gender_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "PERSON:" + c.sensitive.get_span()
        return 'person'

# regex
# quantity
def LF_quantity(c):
    quantity_regex = re.compile([0-999999999999.])
    mgroups = quantity_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "QUANTITY" + c.sensitive.get_span()
        return 'quantity'

# get_right_token
# precentage
def LF_quantity_pct(c):
    if "%"==" ".join(list(get_right_tokens(c[0],window=1))):
        print "QUANTITY" + c.sensitive.get_span()
        return 'quantity'

# regex
# get number with ,000
def LF_quantity_K(c):
    K_regex = re.compile(r','+[000-999])
    mgroups = K_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "QUANTITY" + c.sensitive.get_span()
        return 'quantity'

# regex
# price sign
def LF_quantity_price(c):
    price_regex = re.compile(r'$')
    mgroups = price_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "QUANTITY" + c.sensitive.get_span()
        return 'quantity'

# regex
# password
def LF_password(c):
    password_regex = re.compile("password"|" pw ")
    mgroups = password_regex.findall(c.sensitive.get_span())

    if len(mgroups) > 0:
        print "SOMETHING" + c.sensitive.get_span()
        return 'password'

LFs = [LF_time, LF_date, LF_location, LF_person, LF_org, LF_url, LF_phone, LF_product, LF_event, LF_email, 
        LF_address, LF_job, LF_file, LF_file_before, LF_quantity_K, LF_quantity_pct, LF_person_gender, 
        LF_date_day, LF_date_today_tomorrow, LF_date_month_abb, LF_time_am, LF_time_pm, LF_event_lunch_dinner,
        LF_address_ave_blvd, LF_address_people_before,  LF_person_family, LF_quantity, LF_org_school, LF_org_company,
        LF_org_govern, LF_org_dept, LF_job_1, LF_password, LF_quantity_price, LF_location_gen, LF_location_spe]


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
