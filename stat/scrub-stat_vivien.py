# basic stats of the annotation. 
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import MySQLdb
from sshtunnel import SSHTunnelForwarder
from wordcloud import WordCloud

END_OF_EMAIL = 17112
END_OF_TWEET = 85560

# custom categories which should not be considered as private. 
BLACKLIST = ["ForeignLanguage", "LEGALISSUE", "VersionNumber", "bank", "Currancy", "template", 
            "Currency", "QUALIFICATIONS", "credit", "continent", "Computercode", "animal", "motto", 
            "item", "resource", "resources", "groupname", "datafeed", "action", "task", "LOCATION", 
            "INTERNETTYPE", "directions", "thing", "game", "resume", "Sport", "pronoun"]
            
ORG = ["COMPANYNAME", "group", "CompanyDepartment", "application", "company", "departmentname", "Government",
        "GovenrmentalBody", "Departments", "School", "Hotel", "department", "SCHOOLLOCATIONANDPROGRAM", "government", "website"]
DATE = ["holiday", "day", "year", "dat", "Holiday", "Day"]
FAX_PHONE = ["phonenumber", "documentnumber", "phoneextension", "phoneextention", "PhoneNumber", "telephone", "phone"]
PRODUCT = ["Software", "tvshow", "COMPUTERPROGRAM", "app", "COMPUTERACCESS"]
ADDRESS = ["mailingaddress", "postofficebox"]
URL = ["Webaddress", "networkpath", "filepath", "computerdirectoryname"]
JOB = ["title", "JobTitle",	"credentials","education","policy","degree", "Title", "GovernmentOfficial", "EducationalLevel"]
EVENT = ["meeting"]
OTHER_PLACE = ["Location", "eventlocation", "locationaddress", "RoomNumber", "OfficeNumber", "EVENTLOCATION", "roomnumber"]
FILE = ["File", "filename", "Document", "documentname", "document", "format", "DocumentAttachment", 
        "attachment", "Attachment", "agreementsanddocuments", "doc", "FILENAME", "Documentation", 
        "DocumentName", "Documents", "docfiles"]
TECH_TERM = ["statisticalanalysisterm", "statisticaltechnique"]	
RACE = ["nationality", "Nationality"]

SOCIAL_GROUP = ["groupofpeople", "groupofpersons", "othergroup", "personalinformation"]
QUANTITY = ["quantity", "price", "money", "weight", "length", "balance", "percentage"]

OTHER = ["Stock", "frequentflyernumber", 
        "studytitle", "program", "project", 
        "IDnumber", "ReferenceNumber", 
        "accountnumber", "idnumber", 
        "worktask", "Businessaccount", 
        "attacheddocument", "Continent", 
        "projectname", "unsure", "privatenumber"]
        
PERSON = ["initials", "petname"]

def getSynonym(category):
    
    if category in DATE:
        return "date"
    elif category in PERSON:
        return "person"
    elif category in QUANTITY:
        return "quantity"
    elif category in ORG:
        return 'org'
    elif category in DATE:
        return 'date'
    elif category in FAX_PHONE:
        return "fax_phone"
    elif category in PRODUCT:
        return "product"
    elif category in ADDRESS:
        return "address"
    elif category in URL:
        return 'url'
    elif category in JOB:
        return "job"
    elif category in EVENT:
        return "event"
    elif category in FILE:
        return "file"
    elif category in OTHER_PLACE:
        return "other_place"
    elif category in TECH_TERM:
        return "tech_term"
    elif category in RACE:
        return "race"
    elif category in SOCIAL_GROUP:
        return "social_group"
    elif category in OTHER:
        return "other"
    else:
        return category
        
        
def cleanWord(word):
    return word.replace('\r','').replace('\n','').replace('\'s','')

n_sensitive_im =0
n_sensitive_email = 0
n_sensitive_tweet =0

with SSHTunnelForwarder(
         ('scrubber.yiad.am'),
         ssh_password="v!18pw",
         ssh_username="vivien",
         remote_bind_address=('127.0.0.1', 3306)) as server:

    connection = MySQLdb.connect(host='127.0.0.1',
                           port=server.local_bind_port,
                           user='root',
                           passwd='',
                           db='scrubber')
    cursor=connection.cursor()
    sql= "SELECT id, originalText, annotatedText FROM scrubber_sentence;"

    cursor.execute(sql)
    data=cursor.fetchall()

    p = re.compile('<scrub type=\'\w*?\'>.*?</scrub>',re.IGNORECASE)
    cat=defaultdict(int)
    for line in data:
        originalText = line[1]
        
        if line[2] == "":
            print "originalText::"+originalText
        else:
            print line[2]
       
            
        scrubs = p.findall(line[2])
        # print ("scrubs",scrubs)
        for l in scrubs:
            print("l",l)
            try:
                c = l.split('=')[1].split('\'')[1]
                print ("c",c)
                if c not in BLACKLIST:
                    c = getSynonym (c)
                    word = l.split('>')[1].split('<')[0]
                    print("word",word)
                    cat[ c ] += 1
                    print("0", cat[ c ])
                    with open("../results/"+c, 'a+') as f:
                        print("1")
                        print("clean word:",cleanWord(word)+"\n")
                        f.write(cleanWord(word)+"\n")
                        print("2")
                        

            except Exception as e:
                print(l, e)

        # word cloud
        for category, count in cat.iteritems():

            if line[0] <= END_OF_EMAIL:
                n_sensitive_email += count
            elif line[0] > END_OF_EMAIL and line[0] <= END_OF_TWEET:
                n_sensitive_tweet += count
            else:
                n_sensitive_im += count
            
        
        
    print "number of sensitive info in emails:" + str(n_sensitive_email)
    print "number of sensitive info in tweets:" + str(n_sensitive_tweet)
    print "number of sensitive info in instant messages:" + str(n_sensitive_im)
