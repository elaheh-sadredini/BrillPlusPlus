# A Brill Tagging Rule Generator
# Deyuan Guo and Elaheh Sadredini. CS@UVa. May 2016


import sys
import time
import nltk
from nltk.tbl.template import Template
from nltk.tag.brill import Pos, Word
from nltk.tag import untag, RegexpTagger, BrillTaggerTrainer, UnigramTagger
from nltk.corpus import treebank, brown

def gen_single_feature(word, tag):
    regex = ""
    if word == None and tag == None:
        regex += '[^ ]+'
    else:
        if word != None:
            regex += word
        else:
            regex += '[^ ]*'
        regex += '\/'
        if tag != None:
            regex += tag
        else:
            regex += '[^ ]*'
    return regex

def gen_range_feature(rule, rb, re):
    # Assume there is no overlapped range features
    word = {}
    tag = {}
    word_is_range = False
    tag_is_range = False
    for cond in rule._conditions:
        feature = cond[0]
        value = cond[1]
        if feature.positions[0] >= rb and feature.positions[0] <= re:
            print "Target:", feature, feature.positions, value
            if feature.PROPERTY_NAME == 'Pos':
                if len(feature.positions) > 1:
                    tag_is_range = True
                    for i in feature.positions:
                        tag[i] = value
                else:
                    tag[feature.positions[0]] = value
            elif feature.PROPERTY_NAME == 'Word':
                if len(feature.positions) > 1:
                    word_is_range = True
                    for i in feature.positions:
                        word[i] = value
                else:
                    word[feature.positions[0]] = value
            else:
                assert False
    assert (word_is_range == True and tag_is_range == False) or (word_is_range == False and tag_is_range == True)

    print "Generate range feature:", word, tag
    regex = "("
    for i in range(rb, re + 1):
        for j in range(rb, re + 1):
            if word_is_range == True:
                if word.has_key(j) and j == i:
                    t_word = word[j]
                else:
                    t_word = None
                if tag.has_key(j):
                    t_tag = tag[j]
                else:
                    t_tag = None
                regex += gen_single_feature(t_word, t_tag)
            elif tag_is_range == True:
                if word.has_key(j):
                    t_word = word[j]
                else:
                    t_word = None
                if tag.has_key(j) and j == i:
                    t_tag = tag[j]
                else:
                    t_tag = None
                regex += gen_single_feature(t_word, t_tag)
            if j != re:
                regex += " +"
        if i != re:
            regex += "|"
    regex += ")"

    return regex


# Convert one brill tagging rule to a regex for the AP
# Assume the maximun lookahead is 2
def rule_to_regex(rule, range_l, range_r):
    regex = ''
    report_tag = ''

    verbose = False
    if verbose:
        print "\n========"
        print rule
        print rule.format('str')
        print rule.format('verbose')
        print "Original Tag:", rule.original_tag
        print "Replacement Tag:", rule.replacement_tag
        print "Conditions:", rule._conditions

    # Determine the actual feature range from l to range_r
    lmost = range_r
    for cond in rule._conditions:
        feature = cond[0]
        for i in feature.positions:
            if lmost > i:
                lmost = i
    if verbose:
        print "Leftmost position:", lmost

    # Analyzing the range features in a rule
    # There can be multiple range features in one rule and can be overlapped
    # Example of overlap:
    #   - word0 at range 1-3, word1 at 2
    #   - word0 at range 1-3, word1 at range 2-4
    #   - word0 at range 1-3, tag0 at range 2-4
    #   - *not overlap: word0 at range 1-3, tag0 at 2
    report_tag = rule.replacement_tag
    pos = {}
    word = {}
    range_b = []
    range_e = []
    pos[0] = rule.original_tag
    overlapped = False
    for cond in rule._conditions:
        feature = cond[0]
        value = cond[1]
        if len(feature.positions) == 1: #single
            i = feature.positions[0]
            if feature.PROPERTY_NAME == 'Pos':
                if pos.has_key(i):
                    overlapped = True
                else:
                    pos[i] = value
            elif feature.PROPERTY_NAME == 'Word':
                if word.has_key(i):
                    overlapped = True
                else:
                    word[i] = value
            else:
                print 'unknown feature type', feature.PROPERTY_NAME
                assert False
        else: #range
            if feature.PROPERTY_NAME == 'Pos':
                left = feature.positions[0]
                right = feature.positions[0]
                for i in feature.positions:
                    if pos.has_key(i):
                        overlapped = True
                    else:
                        pos[i] = value
                    if left > i: left = i
                    if right < i: right = i
                range_b.append(left)
                range_e.append(right)
            elif feature.PROPERTY_NAME == 'Word':
                left = feature.positions[0]
                right = feature.positions[0]
                for i in feature.positions:
                    if word.has_key(i):
                        overlapped = True
                    else:
                        word[i] = value
                    if left > i: left = i
                    if right < i: right = i
                range_b.append(left)
                range_e.append(right)
            else:
                print 'unknown feature type', feature.PROPERTY_NAME
                assert False

    if verbose:
        print "Tags:", pos
        print "Words:", word
        print "Range Locations:", range_b, range_e
        print "Overlapped =", overlapped

    if overlapped:
        print "Not support overlapped range feature yet."
        print "Implement it later."
        assert False

    # Generate regex for the AP
    # We can use space or \s for separation
    regex = '/'

    i = lmost
    while i <= range_r:
        regex += ' +'

        # if is a range feature
        find = False
        for j in range(len(range_b)):
            if i == range_b[j]:
                find = True
                range_begin = i
                range_end = range_e[j]
                break
        if find:
            regex += gen_range_feature(rule, range_begin, range_end)
            i = range_end + 1
            continue

        # single features
        cur_word = None
        if word.has_key(i):
            cur_word = word[i]
        cur_tag = None
        if pos.has_key(i):
            cur_tag = pos[i]

        regex += gen_single_feature(cur_word, cur_tag)
        i += 1

    regex += ' /' # reporting

    return regex, report_tag


# determine the range of brill tagging templates
# for reporting pipelining
def get_template_range(templates):
    max_l = 100
    max_r = -100
    for t in templates:
        for f in t._features:
            l = f.positions[0]
            r = f.positions[-1]
            if max_l > l: max_l = l
            if max_r < r: max_r = r
    return max_l, max_r


# Cross validation
def gen_tagging_rules(nrule, my_corpus):
    print "============================================================"

    # Parameters:
    templates = nltk.tag.brill.fntbl37()
    n_rules = nrule
    fold = 5
    ap_freq = 133000000.0
    do_evaluate = False
    input_string_file = "input.txt"
    regex_file = "regex.txt"
    out1 = open(input_string_file, "w+")
    out2 = open(regex_file, "w+")


    # Backoff tagger for the unigram tagger
    regex_tagger = nltk.RegexpTagger(
        [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
         (r'(The|the|A|a|An|an)$', 'AT'),   # articles
         (r'.*able$', 'JJ'),                # adjectives
         (r'.*ness$', 'NN'),                # nouns formed from adjectives
         (r'.*ly$', 'RB'),                  # adverbs
         (r'.*s$', 'NNS'),                  # plural nouns
         (r'.*ing$', 'VBG'),                # gerunds
         (r'.*ed$', 'VBD'),                 # past tense verbs
         (r'.*', 'NN')                      # nouns (default)
        ])

    # k-fold cross validation
    for k in range(fold):
        # Just do 1 fold here for generating regex
        if k != 0: continue

        print "\n== Preparing training data ..."
        if my_corpus == treebank:
            training = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold != k]
            validation = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold == k]
            testing = [x for i, x in enumerate(my_corpus.sents()) if i % fold == k]
        elif my_corpus == brown:
            #training = [x for i, x in enumerate(my_corpus.tagged_sents(categories='news')) if i % fold != k]
            #validation = [x for i, x in enumerate(my_corpus.tagged_sents(categories='news')) if i % fold == k]
            #testing = [x for i, x in enumerate(my_corpus.sents(categories='news')) if i % fold == k]
            training = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold != k]
            validation = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold == k]
            testing = [x for i, x in enumerate(my_corpus.sents()) if i % fold == k]
        else:
            assert False

        #training = [x for i, x in enumerate(my_corpus.tagged_sents(categories='news')) if i % fold != k]
        #validation = [x for i, x in enumerate(my_corpus.tagged_sents(categories='news')) if i % fold == k]
        #testing = [x for i, x in enumerate(my_corpus.sents(categories='news')) if i % fold == k]

        print "\n== Initializing the baseline tagger ..."
        u_gram_tag=nltk.UnigramTagger(training,backoff=regex_tagger)
        baseline = u_gram_tag

        if do_evaluate:
            print "\n== Evaluating testing accuracy of the baseline tagger ..."
            begin = time.time()
            baseline_accuracy = baseline.evaluate(validation)
            end = time.time()
            t_baseline = end - begin
            print "Baseline Testing Time =", t_baseline, "second"
            print "Baseline Accuracy =", baseline_accuracy

        print "\n== Training the Brill tagger ..."
        tt = BrillTaggerTrainer(baseline, templates, trace=3)
        begin = time.time()
        brill_tagger = tt.train(training, max_rules=n_rules)
        end = time.time()
        t_brilltrain = end - begin
        print "Brill Tagger Training Time =", t_brilltrain, "second"
        print "Learn rules:", len(brill_tagger.rules())

        if do_evaluate:
            print "\n== Testing the Brill Tagger ..."
            begin = time.time()
            brill_accuracy = brill_tagger.evaluate(validation)
            end = time.time()
            t_brilltest = end - begin
            print "Brill Tagger Testing Time =", t_brilltest, "second"
            print "Brill Tagger Accuracy =", brill_accuracy
            print "Accuracy improvement:", brill_accuracy - baseline_accuracy

        # analyze the AP running time
        print "\n== Generating AP input string ..."
        baseline_tagged = baseline.tag_sents(testing)
        ap_input = ' ';
        for s in baseline_tagged:
            for w, t in s:
                ap_input += w + '/' + t + ' '
            ap_input += "/ / / "
        total_length = len(ap_input)
        print "AP input bytes:", total_length, "(", total_length / ap_freq, "second)"
        print "Write the input string to", input_string_file, "..."
        out1.write(ap_input)
        out1.write('\n')
        print "The input string is written to", input_string_file


        print "\n== Generating Regex for the AP ..."
        print "Template size:", len(templates)
        range_l, range_r = get_template_range(templates)
        print "Template range:", range_l, range_r
        print "Total rules:", len(brill_tagger.rules())

        for rule in brill_tagger.rules():
            regex, report_tag = rule_to_regex(rule, range_l, range_r)
            print report_tag, ":", regex
            #out2.write(report_tag + " : " + regex + "\n")
            out2.write(regex + "\n")

        print "The regexes are written to", regex_file
        print "Done."
        out1.close()
        out2.close()


# Main
if __name__ == "__main__":
    print "\n    AP POS Tagging Rules Generator.\n    Deyuan Guo & Elaheh Sadredini, May 2016"
    help_msg = """
    Usage: python ap-tagging-rule-gen.py <#rules> <corpus>
    """

    if len(sys.argv) != 3:
        print help_msg
        sys.exit(0)

    num_rules = int(sys.argv[1])
    
    if (sys.argv[2] == "treebank"):
        my_corpus = treebank
    elif (sys.argv[2] == "brown"):
        my_corpus = brown
    else:
        print "Unknown corpus:", sys.argv[2]
        sys.exit(0)
    if num_rules <= 0:
        print help_msg
        sys.exit(0)

    print "\nGenerating", num_rules, "tagging rules based on fnTBL 37 rule templates and the Brown corpus."

    gen_tagging_rules(num_rules, my_corpus)


