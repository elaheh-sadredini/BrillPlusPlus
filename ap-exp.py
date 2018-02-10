# Deyuan & Elaheh. CS@UVa. Apr 2016

import sys
import time
import nltk
from nltk.tbl.template import Template
from nltk.tag.brill import Pos, Word
from nltk.tag import untag, RegexpTagger, BrillTaggerTrainer, UnigramTagger
from nltk.corpus import treebank, brown

# Convert one brill tagging rule to a regex for the AP
# Assume the maximun lookahead is 2
def rule_to_regex(rule, range_l, range_r):
    regex = ''
    report_tag = ''

    verbose = False
    if verbose:
        print "\n", rule
        print rule.format('str')
        print rule.format('verbose')
        print rule.original_tag
        print rule.replacement_tag
        print rule._conditions

    # Special cases:
    # 3\/8/CD
    # Ignore: wsj_0099.pos:[ targeting/VBG|NN equipment/NN ]

    # extract features from rules
    report_tag = rule.replacement_tag
    pos = {}
    word = {}
    pos[0] = rule.original_tag
    for cond in rule._conditions:
        feature = cond[0]
        value = cond[1]
        if feature.PROPERTY_NAME == 'Pos':
            for i in feature.positions:
                pos[i] = value
        elif feature.PROPERTY_NAME == 'Word':
            for i in feature.positions:
                word[i] = value
        else:
            print 'unknown feature type', feature.PROPERTY_NAME
            assert False
    if verbose:
        print pos
        print word

    # determine the actual range from l to range_r
    l = range_l
    for i in range(range_l, range_r):
        if pos.has_key(i) or word.has_key(i):
            l = i
            break

    # generate regex for the AP
    regex = '/'
    for i in range(l, range_r + 1):
        regex += '\s+'
        if not word.has_key(i) and not pos.has_key(i): # skip a word
            regex += '[^\s]+'
        else:
            if word.has_key(i):
                regex += word[i]
            else:
                regex += '[^\s]+'
            regex += '\/'
            if pos.has_key(i):
                regex += pos[i]
            else:
                regex += '[^\s]+'
    regex += '\s/' # reporting

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


# Generate Regex from learned Brill tagging rules
def gen_ap_regex():
    print "============================================================"
    print "Generate Regex from learned Brill tagging rules."
    # Parameters:
    training = my_corpus.tagged_sents()
    templates = nltk.tag.brill.fntbl37()
    n_rules = 30

    # Taggers:
    print "Initializing ..."
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
    u_gram_tag=nltk.UnigramTagger(training,backoff=regex_tagger)
    b_gram_tag=nltk.BigramTagger(training,backoff=u_gram_tag)
    t_gram_tag=nltk.TrigramTagger(training,backoff=b_gram_tag)

    print "Training brill tagger ..."
    tt = BrillTaggerTrainer(t_gram_tag, templates, trace=3)
    brill_tagger = tt.train(training, max_rules=n_rules)
    print "Training finished."

    print "Template size:", len(templates)
    range_l, range_r = get_template_range(templates)
    print "Template range:", range_l, range_r
    print "Total rules:", len(brill_tagger.rules())
    print "Generating Regex for the AP ..."

    for rule in brill_tagger.rules():
        regex, report_tag = rule_to_regex(rule, range_l, range_r)
        print report_tag, ":", regex

    print "Done."


# Cross validation
def test_regex_tagger(tagger_id):
    print "============================================================"
    print "Cross Validation on corpus."

    # Parameters:
    fold = 5
    ap_freq = 133000000.0

    b_accuracy = []
    b_time = []

    regex_tagger_1 = nltk.RegexpTagger(
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

    regex_tagger_2 = nltk.RegexpTagger(
        [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
         (r'(The|the|A|a|An|an)$', 'AT'),   # articles
         (r'.*able$', 'JJ'),                # adjectives
         (r'.*ness$', 'NN'),                # nouns formed from adjectives
         (r'.*ly$', 'RB'),                  # adverbs
         (r'.*s$', 'NNS'),                  # plural nouns
         (r'.*ing$', 'VBG'),                # gerunds
         (r'.*ed$', 'VBD'),                 # past tense verbs
         (r'[^\s]+-[^\s]+$', 'T1'),
         (r'[^\s]*[A-Z][^\s]*$', 'T2'),
         (r'[^\s]*([a-zA-Z][0-9]|[0-9][a-zA-Z])[^\s]*$', 'T3'),
         (r'[^\s]*([a-zA-Z])\1\1[^\s]*$', 'T4'),
         (r'.*', 'NN')                      # nouns (default)
        ])

    for k in range(fold):
        if my_corpus == treebank:
            training = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold != k]
            validation = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold == k]
            testing = [x for i, x in enumerate(my_corpus.sents()) if i % fold == k]
        elif my_corpus == brown:
            training = [x for i, x in enumerate(my_corpus.tagged_sents(categories='news')) if i % fold != k]
            validation = [x for i, x in enumerate(my_corpus.tagged_sents(categories='news')) if i % fold == k]
            testing = [x for i, x in enumerate(my_corpus.sents(categories='news')) if i % fold == k]
        else:
            assert False

        print "\n\nFold", k, "Initializing ..."
        if tagger_id == 1:
            baseline = regex_tagger_1
        elif tagger_id == 2:
            baseline = regex_tagger_2
        else:
            assert False

        print "Evaluating testing accuracy ..."
        begin = time.time()
        baseline_accuracy = baseline.evaluate(validation)
        end = time.time()
        t1 = end - begin
        print "Baseline Testing Time =", t1, "second"
        print "Baseline Accuracy =", baseline_accuracy

        b_accuracy.append(baseline_accuracy)
        b_time.append(t1)


    print "\n\nCross Validation Results (baseline):"
    print "Average Accuracy:", sum(b_accuracy)/len(b_accuracy)
    print "Average Time:", sum(b_time)/float(len(b_time))



# Cross validation
def test_cross_validation(btagger, tmpl, nrule):
    print "============================================================"
    print "Cross Validation on corpus."

    # Parameters:
    if tmpl == 'brill24':
        templates = nltk.tag.brill.brill24()
    elif tmpl == 'fntbl37':
        templates = nltk.tag.brill.fntbl37()
    else:
        assert False
    n_rules = nrule
    fold = 5
    ap_freq = 133000000.0

    accuracy1 = []
    accuracy2 = []
    brill_time = []
    ap_time = []
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
    for k in range(fold):
        if my_corpus == treebank:
            training = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold != k]
            validation = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold == k]
            testing = [x for i, x in enumerate(my_corpus.sents()) if i % fold == k]
        elif my_corpus == brown:
            training = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold != k]
            validation = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold == k]
            testing = [x for i, x in enumerate(my_corpus.sents()) if i % fold == k]
            #training = [x for i, x in enumerate(my_corpus.tagged_sents(categories='news')) if i % fold != k]
            #validation = [x for i, x in enumerate(my_corpus.tagged_sents(categories='news')) if i % fold == k]
            #testing = [x for i, x in enumerate(my_corpus.sents(categories='news')) if i % fold == k]
        else:
            assert False

        print "\n\nFold", k, "Initializing ...", tmpl, ", ", btagger
        if btagger == 'r':
            baseline = regex_tagger
        elif btagger == 'u':
            u_gram_tag=nltk.UnigramTagger(training,backoff=regex_tagger)
            baseline = u_gram_tag
        elif btagger == 'b':
            u_gram_tag=nltk.UnigramTagger(training,backoff=regex_tagger)
            b_gram_tag=nltk.BigramTagger(training,backoff=u_gram_tag)
            baseline = b_gram_tag
        elif btagger == 't':
            u_gram_tag=nltk.UnigramTagger(training,backoff=regex_tagger)
            b_gram_tag=nltk.BigramTagger(training,backoff=u_gram_tag)
            t_gram_tag=nltk.TrigramTagger(training,backoff=b_gram_tag)
            baseline = t_gram_tag
        elif btagger == 's': # stanford
            stagger = nltk.tag.stanford.StanfordPOSTagger(stanford_path+'models/english-bidirectional-distsim.tagger', stanford_path+'stanford-postagger.jar')
            baseline = stagger
        else:
            assert False

        print "Evaluating testing accuracy ..."
        begin = time.time()
        baseline_accuracy = baseline.evaluate(validation)
        end = time.time()
        print "Baseline Testing Time =", end - begin, "second"
        t1 = end - begin
        print "Baseline Accuracy =", baseline_accuracy

        tt = BrillTaggerTrainer(baseline, templates, trace=3)
        print "Training Brill tagger ..."
        begin = time.time()
        brill_tagger = tt.train(training, max_rules=n_rules)
        end = time.time()
        print "Brill Tagger Training Time =", end - begin, "second"
        print "Find rules:", len(brill_tagger.rules())

        print "Testing ..."
        begin = time.time()
        brill_accuracy = brill_tagger.evaluate(validation)
        end = time.time()
        print "Brill Tagger Test Time =", end - begin, "second"
        t2 = end - begin
        print "Brill Tagger Accuracy =", brill_accuracy
        print "Accuracy improvement:", brill_accuracy - baseline_accuracy

        accuracy1.append(baseline_accuracy)
        accuracy2.append(brill_accuracy)
        brill_time.append(end - begin)

        # analyze the AP running time
        print "Generating AP input string ..."
        baseline_tagged = baseline.tag_sents(testing)
        ap_input = '';
        for s in baseline_tagged:
            for w, t in s:
                ap_input += w + '/' + t + ' '
            ap_input += "./-NONE- ./-NONE- ./-NONE- "
        total_length = len(ap_input)
        print "AP input bytes:", total_length, "(", total_length / ap_freq, "second)"
        print "Speedup:", (t2 - t1) / (total_length / ap_freq)
        #print ap_input
        ap_time.append(total_length / ap_freq)

    print "\n\nCross Validation Results (baseline):"
    print accuracy1
    print "\nCross Validation Results (brill):"
    print accuracy2
    print "Brill running time:"
    print brill_time
    print "AP running time:"
    print ap_time
    print "Average Accuracy:", sum(accuracy1)/len(accuracy1), sum(accuracy2)/len(accuracy2)
    print "Average Accuracy:", sum(accuracy2)/len(accuracy2)

    avg_brill_time = sum(brill_time) / float(len(brill_time))
    avg_ap_time = sum(ap_time) / float(len(ap_time))
    print "Average Brill Time:", avg_brill_time
    print "Average AP Time:", avg_ap_time
    print "Speedup (ap over brill):", avg_brill_time / avg_ap_time


# Cross validation for NLTK baseline taggers
def test_baseline_cv(btagger):
    print "============================================================"
    print "Cross Validation on corpus."

    fold = 5

    train_acc = {}
    test_acc = {}
    test_time = {}
    train_acc[btagger] = []
    test_acc[btagger] = []
    test_time[btagger] = []
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
    for k in range(fold):
        if my_corpus == treebank:
            training = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold != k]
            validation = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold == k]
            testing = [x for i, x in enumerate(my_corpus.sents()) if i % fold == k]
        #elif my_corpus == brown:
            #training = [x for i, x in enumerate(my_corpus.tagged_sents(categories='news')) if i % fold != k]
            #validation = [x for i, x in enumerate(my_corpus.tagged_sents(categories='news')) if i % fold == k]
            #testing = [x for i, x in enumerate(my_corpus.sents(categories='news')) if i % fold == k]
        elif my_corpus == brown:
            training = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold != k]
            validation = [x for i, x in enumerate(my_corpus.tagged_sents()) if i % fold == k]
            testing = [x for i, x in enumerate(my_corpus.sents()) if i % fold == k]
        else:
            assert False


        print "\n\nFold", k, "Initializing ..."

        if btagger == 'r':
            print "RegexpTagger"
            baseline = regex_tagger
        elif btagger == 'u':
            u_gram_tag=nltk.UnigramTagger(training,backoff=regex_tagger)
            print "UnigramTagger"
            baseline = u_gram_tag
        elif btagger == 'b':
            u_gram_tag=nltk.UnigramTagger(training,backoff=regex_tagger)
            b_gram_tag=nltk.BigramTagger(training,backoff=u_gram_tag)
            print "BigramTagger"
            baseline = b_gram_tag
        elif btagger == 't':
            u_gram_tag=nltk.UnigramTagger(training,backoff=regex_tagger)
            b_gram_tag=nltk.BigramTagger(training,backoff=u_gram_tag)
            t_gram_tag=nltk.TrigramTagger(training,backoff=b_gram_tag)
            print "TrigramTagger"
            baseline = t_gram_tag
        elif btagger == 's': # stanford
            stagger = nltk.tag.stanford.StanfordPOSTagger(stanford_path+'models/english-bidirectional-distsim.tagger', stanford_path+'stanford-postagger.jar')
            baseline = stagger
        else:
            assert False

        print "Evaluating training accuracy ..."
        train_accuracy = baseline.evaluate(training)
        print "Baseline Train Accuracy =", train_accuracy
        begin = time.time()
        baseline_accuracy = baseline.evaluate(validation)
        end = time.time()
        print "Baseline Test Accuracy =", baseline_accuracy
        print "Baseline Test Time =", end - begin, "second"
        t1 = end - begin

        train_acc[btagger].append(train_accuracy)
        test_acc[btagger].append(baseline_accuracy)
        test_time[btagger].append(t1)

    print "\n\nCross Validation Results (baseline):"
    print train_acc
    print test_acc
    print test_time

    print "Average Accuracy:", sum(train_acc[btagger])/len(train_acc[btagger]), sum(test_acc[btagger])/len(test_acc[btagger])
    print "Average Test Time:", sum(test_time[btagger])/len(test_time[btagger])



# Main
if __name__ == "__main__":
    print "\n    POS Tagging Experiments.\n    Deyuan & Elaheh, Apr 2016"
    help_msg = """
    Usage: python ap-exp.py <corpus> <#test>
           <corpus> = treebank or brown
           <#test> = 100 to k
    Example: python ap-exp.py treebank 100
    """

    if len(sys.argv) != 3:
        print help_msg
        sys.exit(0)

    if (sys.argv[1] == "treebank"):
        my_corpus = treebank
    elif (sys.argv[1] == "brown"):
        my_corpus = brown

    else:
        print "Unknown corpus:", sys.argv[1]
        sys.exit(0)

    test_id = int(sys.argv[2])

    stanford_path = '/if10/dg7vp/Lab/ap-tagging/stanford/stanford-postagger-2015-12-09/'

    print "\nCorpus =", sys.argv[1], " Test =", test_id
    if test_id == 100:
        test_baseline_cv("r")
        test_baseline_cv("u")
        test_baseline_cv("b")
        test_baseline_cv("t")
    elif test_id == 101:
        test_baseline_cv("s")
    elif test_id == 102:
        test_cross_validation('u', 'brill24', 500)
    elif test_id == 103:
        test_cross_validation('u', 'fntbl37', 500)
    elif test_id == 104:
        test_cross_validation('u', 'brill24', 500)
        test_cross_validation('u', 'fntbl37', 500)
        test_cross_validation('b', 'brill24', 500)
        test_cross_validation('b', 'fntbl37', 500)
        test_cross_validation('t', 'brill24', 500)
        test_cross_validation('t', 'fntbl37', 500)
    elif test_id == 105:
        test_cross_validation('u', 'fntbl37', 100)
        test_cross_validation('u', 'fntbl37', 200)
        test_cross_validation('u', 'fntbl37', 300)
        test_cross_validation('u', 'fntbl37', 400)
        test_cross_validation('u', 'fntbl37', 500)

    elif test_id == 111: # test regex tagger with more rules
        test_regex_tagger(1)
        test_regex_tagger(2)

    elif test_id == 121: # Generate Regex from learned Brill tagging rules
        gen_ap_regex()




