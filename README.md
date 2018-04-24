# Brill-
Rule-based part-of-speech tagging with different baseline taggers and different set of rules.


## Required Packages:
python 2.7 and NLTK 

`python`

`>>> import nltk`

`>>> nltk.download('punkt')`

`>>> nltk.download('treebank')`

`>>> nltk.download('brown')`

`>>> nltk.download('averaged_perceptron_tagger')`


## Rule Generation:
 
`python ap-tagging-rule-gen.py <#rules> <corpus = 'brown' or 'treebank'>`

The generated rules, which are converted to regular expressions, are witten in "regex.txt" file. 

The input to the automata is written in "input.txt".


## Rule to MNRL format:

To convert the rules to MNRL format, use [hscompile](https://github.com/kevinaangstadt/hscompile) with the following commad:
 
`pcre2mnrl regext.txt regex.mnrl`


