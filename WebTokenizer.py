# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:48:00 2018

@author: Ali
"""
###############################################################################
# The following strings are components in the regular expression
# that is used for tokenizing. It's important that phone_number
# appears first in the final regex (since it can contain whitespace).
# It also could matter that tags comes after emoticons, due to the
# possibility of having text like
#
#     <:| and some text >:)
#
# Most importantly, the final element should always be last, since it
# does a last ditch whitespace-based tokenization of whatever is left.
###############################################################################

import re
import html.entities

# This particular element is used in a couple ways, so we define it
# with a name:
emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""
# Twitter symbols/cashtags:  # Added by awd, 20140410.
# Based upon Twitter's regex described here: <https://blog.twitter.com/2013/symbols-entities-tweets>.
cashtag_string = r"""(?:\$[a-zA-Z]{1,6}([._][a-zA-Z]{1,2})?)"""

# The components of the tokenizer:
regex_strings = (
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?            
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?    
      \d{3}          # exchange
      [\-\s.]*   
      \d{4}          # base
    )"""
    ,
    # Emoticons:
    emoticon_string
    ,
    # HTML tags:
    r"""(?:<[^>]+>)"""
    ,
    # URLs:
    r"""(?:http[s]?://t.co/[a-zA-Z0-9]+)"""
    ,
    # Twitter username:
    r"""(?:@[\w_]+)"""
    ,
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    ,
    # Twitter symbols/cashtags:
    cashtag_string
    ,
    # Remaining word types:
    r"""
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
)

dictionary_symbols = [  # list of all symbols
    '.',
    '..',
    '...',
    '!',
    '?',
    ';',
    ':',
    ',',
    ')',
    '(',
    ']',
    '[',
    '-',
    '_',
    '*',
    '+',
    '\'',
    '"',
    '\\',
    '>',
    '<',
    '^']

###############################################################################
# This is the core tokenizing regex:

word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)

# The emoticon and cashtag strings get their own regex so that we can preserve case for them as needed:
emoticon_re = re.compile(emoticon_string, re.VERBOSE | re.I | re.UNICODE)
cashtag_re = re.compile(cashtag_string, re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
html_entity_digit_re = re.compile(r"&#\d+;")
html_entity_alpha_re = re.compile(r"&\w+;")
amp = "&amp;"


###############################################################################


class WebTokenizer(object):
    def __init__(self, *, preserve_case: bool = False):
        self.preserve_case = preserve_case

    def tokenize(self, tweet: str) -> list:
        """
        Argument: tweet -- any string object.
        Value: a tokenized list of strings; concatenating this list returns the original string if preserve_case=True
        """
        # Fix HTML character entitites:
        tweet = self._html2unicode(tweet)
        # Tokenize:
        matches = word_re.finditer(tweet)
        if self.preserve_case:
            listOfTokens = [match.group() for match in matches]
        listOfTokens = [self._normalize_token(match.group()) for match in matches]
        listOfTokens = self._removeCustomizeChracters(listOfTokens)

        return listOfTokens

    @staticmethod
    def _normalize_token(token: str) -> str:

        if emoticon_re.search(token):
            # Avoid changing emoticons like :D into :d
            return token
        if token.startswith('$') and cashtag_re.search(token):
            return token.upper()
        return token.lower()

    @staticmethod
    def _html2unicode(tweet: str) -> str:
        """
        Internal method that seeks to replace all the HTML entities in
        tweet with their corresponding unicode characters.
        """
        # First the digits:
        ents = set(html_entity_digit_re.findall(tweet))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    tweet = tweet.replace(ent, chr(entnum))
                except:
                    pass
        # Now the alpha versions:
        ents = set(html_entity_alpha_re.findall(tweet))
        ents = filter((lambda x: x != amp), ents)
        for ent in ents:
            entname = ent[1:-1]
            # print(entname)
            try:
                tweet = tweet.replace(ent, chr(html.entities.name2codepoint[entname]))
            except:
                pass
            tweet = tweet.replace(amp, " and ")
        return tweet

    @staticmethod
    def _removeCustomizeChracters(listOfTokens: list) -> list:
        for token in listOfTokens:
            if (token in dictionary_symbols):
                listOfTokens.remove(token)

        for token in listOfTokens:
            if (token.isdigit()):
                listOfTokens.remove(token)

        for token in listOfTokens:
            check = re.sub(r"\b\d+\b", "", token)
            if (check == ""):
                listOfTokens.remove(token)

        return listOfTokens