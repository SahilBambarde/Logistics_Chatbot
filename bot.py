 # Meet Pybot: your friend

import nltk
import warnings
warnings.filterwarnings("ignore")
# nltk.download() # for downloading packages
#import tensorflow as tf
import numpy as np
import random
import string # to process standard python strings

f=open('nlp python answer finals.txt','r',errors = 'ignore')
m=open('modules pythons.txt','r',errors = 'ignore')
checkpoint = "./chatbot_weights.ckpt"
#session = tf.InteractiveSession()
#session.run(tf.global_variables_initializer())
#saver = tf.train.Saver()
#saver.restore(session, checkpoint)

raw=f.read()
rawone=m.read()
raw=raw.lower()# converts to lowercase
rawone=rawone.lower()# converts to lowercase
nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words
sent_tokensone = nltk.sent_tokenize(rawone)# converts to list of sentences 
word_tokensone = nltk.word_tokenize(rawone)# converts to list of words


sent_tokens[:2]
sent_tokensone[:2]

word_tokens[:5]
word_tokensone[:5]

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

Introduce_Ans = ["My name is Logistic PyBot.","My name is Logistic PyBot you can called me Logistic Bot.","Im Logistic PyBot :) ","My name is Logistic PyBot and my nickname is pi and i am happy to help you :) "]
GREETING_INPUTS = ("hello", "hi","hiii","hii","hiiii","hiiii", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["Hello I am glad! Please ask me your questions about products and services"]
GREETING_INPUTS2 = ("Thanks","thanks","thankyou","thank you")
GREETING_RESPONSES2 = ["Thanks. Have a nice day!"]

Basic = ("when my package deliver","when my package be delivered?","when my package be delivered ?","when my package will deliver","when my package will arrive")
Basic_1 = "Do you know your order number ?"

Basic_Om = ("how do i start shipping online","how can i start shipping")
Basic_AnsM = "If you are a FedEx account holder, you access FedEx digital solutions by creating a fedex.com User ID and Password"
cluster_1 = ("yes","yes i know")
cluster_Ans = ["Great, Please tell me your order number and I will retrieve your tracking details"]
tension_1 = ("1234","12345","123","123456","order details","tell me my order details","tell my order details")
tension_Ans ="64 GB SandDisk Pendrive Rs.820 Will be delivered on this Tuesday","64 GB SandDisk Pendrive Will be delivered on Wednesday","64 GB SandDisk Pendrive After 3 days","64 GB SandDisk Pendrive Will be delivered within 4-7 days","64 GB SandDisk Pendrive Will be delivered on next Friday"
cough_1 = ("can you track my product","track my product")
cough_Ans ="Your Product 64 GB SandDisk Pendrive Rs.820 has left from mumbai facility today 5:39 am will arrive Today at 7:00 pm"
robot_1 = ("are you robot","are you a robot","You are robot","robot")
robot_Ans ="Yes I am a Robot. How may I help You"
covid_1 = ("change address","can you deliver my product at my friend's address","i wish to change my address","change my address","deliver package at new address","change address","update address","i want to change my address","i want to update my address")
covid_Ans = "Please tell me your new address"

covidtreat_1 = ("nerul","sector","building")
covidtreat_Ans = ["Your address updated successfully"]

book_1 = ("place my order","how can i place my order","how can i order something","i want to order something","i wish to buy something","i wish to place my order","i want to buy something")
book_Ans = "Please visit our website and there you can place your order. Thank You!"
cancel_1 = ("cancel","i want to cancel my order no.123","i dont want product no.123","i wish to cancel my order no.123")
cancel_Ans =["Your order no.123 has cancel successfully"]
return_1 = ("return","i want to return my order no.123","i want to return my product","how i can return my product","how can i replace my product","i wish to replace my order no.123")
return_Ans =["Please visit our website and there you can view your order and just click on 'Return " \
            "my Order' button and then refund amount will send to your bank account number 81XXXXXXXX4589"]
replace_1 = ("replace","i want to replace my order no.123","i want to replace my product","how i can replace my product","how can i return my product","i wish to return my order no.123")
replace_Ans =["Please visit our website and there you can view your package and just click on 'Replace my product' button."]
damage_1 = ("defective","damage","i receive defective product","i receive damage product","i got defective product","uct","my product is not working properly")
damage_Ans =["Will you like to REPLACE your product with new one or want to RETURN it and get refund"]
early_1 = ("early","can i get my product early","i want my product to deliver early","i want early delivery")
early_Ans =["Sure Sir. But for that you will have pay extra charge of Rs.40"]
ok_1 = ("okay","ok","make my delivery early","make it","deliver it early")
ok_Ans ="Thank You. Your Product will arrive 5 days early on tomorrow before 9:00 pm"
bs_1 = ("budget","budget phone","under")
bs_Ans =["Samsung Galaxy M31s = Rs 18499, Oppo F17 = Rs 16499"]
bsp_1 = ("best phone","best","best smartphone")
bsp_Ans =["Samsung Galaxy S20 Ultra +"]
discount_1 = ("discount","offer","exchange","discounts","offers")
discount_Ans =["Yes During Diwali we have upto 80% off on SmartPhones as well as on other electronics. And Exchange"
               " offer also start from 12th November and Ends on 16th November"]


# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def greeting2(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS2:
            return random.choice(GREETING_RESPONSES2)

def basic(sentence):
    for word in Basic:
        if sentence.lower() == word:
            return Basic_1
def basicM(sentence):
    for word in Basic_Om:
        if sentence.lower() == word:
            return Basic_AnsM

def cluster(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in cluster_1:
            return random.choice(cluster_Ans)


# Checking for Tension
def tension(sentence):
    for word in tension_1:
        if sentence.lower() == word:
            return random.choice(tension_Ans)

# Checking for Cough
def cough(sentence):
    for word in cough_1:
        if sentence.lower() == word:
            return cough_Ans

# Checking for Robot
def robot(sentence):
    for word in robot_1:
        if sentence.lower() == word:
            return robot_Ans

def covid(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in covid_1:
        if sentence.lower() == word:
            return covid_Ans

def covidtreat(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in covidtreat_1:
            return random.choice(covidtreat_Ans)

def book(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in book_1:
        if sentence.lower() == word:
            return book_Ans

def cancel(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in cancel_1:
            return random.choice(cancel_Ans)

def return1(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in return_1:
            return random.choice(return_Ans)

def replace(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in replace_1:
            return random.choice(replace_Ans)

def damage(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in damage_1:
            return random.choice(damage_Ans)

def early(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in early_1:
            return random.choice(early_Ans)



def ok(sentence):
    for word in ok_1:
        if sentence.lower() == word:
            return ok_Ans

def bs(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in bs_1:
            return random.choice(bs_Ans)

def bsp(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in bsp_1:
            return random.choice(bsp_Ans)

def discount(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in discount_1:
            return random.choice(discount_Ans)


# Checking for Introduce
def IntroduceMe(sentence):
    return random.choice(Introduce_Ans)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response
      
# Generating response
def responseone(user_response):
    robo_response=''
    sent_tokensone.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokensone)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"Please ask me a question about Products and Services"
        return robo_response
    else:
        robo_response = robo_response+sent_tokensone[idx]
        return robo_response


def chat(user_response):
    user_response=user_response.lower()
    keyword = " module "
    keywordone = " module"
    keywordsecond = "module "
    
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            #print("ROBO: You are welcome..")
            return "You are welcome.."
        elif (basic(user_response) != None):
            return basic(user_response)
        elif(basicM(user_response)!=None):
            return basicM(user_response)
        elif (cluster(user_response) != None):
            return cluster(user_response)
        elif (tension(user_response) != None):
            return tension(user_response)
        elif (cough(user_response) != None):
            return cough(user_response)
        elif (robot(user_response) != None):
            return robot(user_response)

        elif (covidtreat(user_response) != None):
            return covidtreat(user_response)
        elif (book(user_response) != None):
            return book(user_response)
        elif (cancel(user_response) != None):
            return cancel(user_response)
        elif (return1(user_response) != None):
            return return1(user_response)
        elif (replace(user_response) != None):
            return replace(user_response)
        elif (damage(user_response) != None):
            return damage(user_response)
        elif (early(user_response) != None):
            return early(user_response)
        elif (ok(user_response) != None):
            return ok(user_response)
        elif (bs(user_response) != None):
            return bs(user_response)
        elif (discount(user_response) != None):
            return discount(user_response)

        else:
            if(user_response.find(keyword) != -1 or user_response.find(keywordone) != -1 or user_response.find(keywordsecond) != -1):
                #print("ROBO: ",end="")
                #print(responseone(user_response))
                return responseone(user_response)
                sent_tokensone.remove(user_response)
            elif(greeting(user_response)!=None):
                #print("ROBO: "+greeting(user_response))
                return greeting(user_response)
            elif (greeting2(user_response) != None):
                # print("ROBO: "+greeting(user_response))
                return greeting2(user_response)
            elif (covid(user_response) != None):
                return covid(user_response)
            elif(user_response.find("your name") != -1 or user_response.find(" your name") != -1 or user_response.find("your name ") != -1 or user_response.find(" your name ") != -1):
                return IntroduceMe(user_response)

            elif (bsp(user_response) != None):
                return bsp(user_response)
            elif (covid(user_response) != None):
                return covid(user_response)
            else:
                #print("ROBO: ",end="")
                #print(response(user_response))
                return response(user_response)
                sent_tokens.remove(user_response)
                
    else:
        flag=False
        #print("ROBO: Bye! take care..")
        return "Bye! take care.."
        
        

