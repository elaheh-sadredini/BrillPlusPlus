# Test the TnT PoS Tagger in NLTK
# Deyuan Guo, Feb 04, 2018

import sys
import time
from nltk.corpus import treebank, brown
from nltk.tag import tnt

# Test the TnT PoS tagger with 5-fold CV
def run_test(my_corpus):
    if my_corpus == treebank:
        print 'Corpus Info:'
        print '  Corpus: treebank'
        print '  Tagged Sents:', len(my_corpus.tagged_sents())
        print '  Tagged Words:', len(my_corpus.tagged_words())
        my_tagged_sents = my_corpus.tagged_sents()
        my_sents = my_corpus.sents()
    elif my_corpus == brown:
        print 'Corpus Info:'
        print '  Corpus: brown'
        print '  Tagged Sents:', len(my_corpus.tagged_sents())
        print '  Tagged Words:', len(my_corpus.tagged_words())
        #print '  Tagged Sents (news):', len(my_corpus.tagged_sents(categories='news'))
        #print '  Tagged Words (news):', len(my_corpus.tagged_words(categories='news'))
        #my_tagged_sents = my_corpus.tagged_sents(categories='news')
        #my_sents = my_corpus.sents(categories='news')
        
        print '  Tagged Sents :', len(my_corpus.tagged_sents())
        print '  Tagged Words :', len(my_corpus.tagged_words())
        my_tagged_sents = my_corpus.tagged_sents()
        my_sents = my_corpus.sents()
    else:
        return

    fold = 5
    print 'Performing', fold, 'fold cross validation on corpus ...'
    train_accuracy = []
    test_accuracy = []
    train_runtime = []
    test_runtime = []

    for k in range(fold):
        train_data = [x for i, x in enumerate(my_tagged_sents) if i % fold != k]
        validation_data = [x for i, x in enumerate(my_tagged_sents) if i % fold == k]
        #test_data = [x for i, x in enumerate(my_sents) if i % fold == k]

        print 'Fold', k, ' has', len(train_data), 'train sentences and', len(validation_data), 'test sentences'
        tnt_pos_tagger = tnt.TnT()

        begin = time.time()
        tnt_pos_tagger.train(train_data)
        end = time.time()
        train_acc = tnt_pos_tagger.evaluate(train_data)
        train_accuracy.append(train_acc)
        train_runtime.append(end - begin)
        print '  Train accuracy =', train_acc, ' runtime =', end - begin

        begin = time.time()
        test_acc = tnt_pos_tagger.evaluate(validation_data)
        end = time.time()
        test_accuracy.append(test_acc)
        test_runtime.append(end - begin)
        print '  Test accuracy =', test_acc, ' runtime =', end - begin

    print 'Results:'
    print '%15s %15s %15s %15s %15s' % ('Fold', 'Train-Accuracy', 'Train-Runtime', 'Test-Accuracy', 'Test-Runtime')
    for k in range(fold):
        print '%15d %15.3f%% %15.5f %15.3f%% %15.5f' % (k, train_accuracy[k] * 100, train_runtime[k], test_accuracy[k] * 100, test_runtime[k])

    avg_train_acc = sum(train_accuracy)/len(train_accuracy)
    avg_train_runtime = sum(train_runtime)/len(train_runtime)
    avg_test_acc = sum(test_accuracy)/len(test_accuracy)
    avg_test_runtime = sum(test_runtime)/len(test_runtime)
    print '%15s %15.3f%% %15.5f %15.3f%% %15.5f' % ('Average', avg_train_acc * 100, avg_train_runtime, avg_test_acc * 100, avg_test_runtime)
    return


# Main
if __name__ == '__main__':
    print 'Testing the TnT PoS Tagger in NLTK.'
    help_msg = 'Usage: test_tnt_tagger.py <treebank|brown>'

    if len(sys.argv) != 2:
        print help_msg
        sys.exit(0)

    if sys.argv[1] == 'treebank':
        my_corpus = treebank
    elif sys.argv[1] == 'brown':
        my_corpus = brown
    else:
        print 'Unknown corpus:', sys.argv[1]
        sys.exit(0)

    run_test(my_corpus)


