import nltk
from preprocess import doc_ranker_simple,doc_ranker
from collections import defaultdict
import codecs
import pickle

root = '/home1/w/wenpeng/dataset/FEVER/'

def transfer_wordlist_2_idlist_with_maxlen_InTest(token_list, vocab_map, maxlen):
    '''
    use in test, so not allow vocab_map to increase
    '''
    idlist=[]
    for word in token_list:
        position = word.find('-')
        if position<0:
#             if word not in string.punctuation:
#                 word =  word.translate(None, string.punctuation)
            id=vocab_map.get(word)
            if id is not None: # if word was not in the vocabulary
                idlist.append(id)
        else:
            subwords = word.split('-')
            for subword in subwords:
#                 if subword not in string.punctuation:
#                     subword =  subword.translate(None, string.punctuation)
                id=vocab_map.get(subword)
                if id is not None: # if word was not in the vocabulary
                    idlist.append(id)

    mask_list=[1.0]*len(idlist) # mask is used to indicate each word is a true word or a pad word
    pad_size=maxlen-len(idlist)
    if pad_size>0:
        idlist=[0]*pad_size+idlist
        mask_list=[0.0]*pad_size+mask_list
    else: # if actual sentence len is longer than the maxlen, truncate
        idlist=idlist[:maxlen]
        mask_list=mask_list[:maxlen]
    return idlist, mask_list

def claim_2_all_sents_of_top5_articles(claim,title2sentlist, title2wordlist):
    '''
    load wiki
    '''


    # title2sentlist={}
    # title2wordlist = {}
    # # word2titlelist=defaultdict(list)
    # readwiki = codecs.open(root+'wiki_title2sentlist.txt' ,'r', 'utf-8')
    # wiki_co = 0
    # for line in readwiki:
    #     parts = line.strip().split('\t')
    #     title = parts[0]
    #     title2sentlist[title] = parts[1:]
    #     title_wordlist = title.replace('-LRB-','').replace('-RRB-','').split('_')
    #     title2wordlist[title] = title_wordlist+line.strip().split()
    #
    #     # title_vocab = set(title_wordlist)
    #     # for word in title_vocab:
    #     #     word2titlelist[word].append(title)
    #     wiki_co+=1
    #     if wiki_co % 1000000 ==0:
    #         print('wiki_co....', wiki_co)
    # readwiki.close()
    # print('wiki pages loaded over, totally ', len(title2wordlist), ' pages')
    # with open(root+'title2sentlist.p', 'wb') as fp:
    #     pickle.dump(title2sentlist, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(root+'title2wordlist.p', 'wb') as fp:
    #     pickle.dump(title2wordlist, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(root+'word2titlelist.p', 'wb') as fp:
    #     pickle.dump(word2titlelist, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # print('three dictionaries stored over')
    # exit(0)
    # with open(root+'title2sentlist.p', 'rb') as fp:
    #     title2sentlist = pickle.load(fp)
    # with open(root+'title2wordlist.p', 'rb') as fp:
    #     title2wordlist = pickle.load(fp)
    # with open(root+'word2titlelist.p', 'rb') as fp:
    #     word2titlelist = pickle.load(fp)
    # print('wiki load over...')

    '''
    claim to top 5 docs
    '''
    # doc_list = doc_ranker_simple(claim,word2titlelist, title2wordlist)
    doc_list, _ = doc_ranker(claim, title2wordlist)
    print('wiki articles rank over...')
    # print(doc_list)
    # exit(0)
    '''
    top5 docs to all sentence cand
    '''
    sent_cand_list = []
    for doc in doc_list:
        doc2sents = title2sentlist.get(doc)
        if doc2sents is not None:
            for i, sent in enumerate(doc2sents):
                if len(sent.strip()) == 0:
                    continue
                else:
                    sent_cand_list.append(sent)
    if len(sent_cand_list) == 0:
        print('empty evidence candidates, exit')
        exit(0)
    return sent_cand_list

def claim_input_2_theano_input(claim, word2id, claim_len, sent_len, cand_size,title2sentlist, title2wordlist):
    sent_cand_list = claim_2_all_sents_of_top5_articles(claim,title2sentlist, title2wordlist)
    claim_wordlist=nltk.word_tokenize(claim)
    claim_idlist, claim_masklist=transfer_wordlist_2_idlist_with_maxlen_InTest(claim_wordlist, word2id, claim_len)

    true_cand_size = len(sent_cand_list)
    if true_cand_size > cand_size:
        sent_cand_list = sent_cand_list[:cand_size]
    else:
        append_size = cand_size - true_cand_size
        sent_cand_list = sent_cand_list+sent_cand_list[-1:]*append_size
    assert len(sent_cand_list) == cand_size
    sent_ins_ids = []
    sent_ins_mask = []
    for sent in sent_cand_list:
        sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen_InTest(sent.split(), word2id, sent_len)
        sent_ins_ids.append(sent_idlist)
        sent_ins_mask.append(sent_masklist)

    return claim_idlist, claim_masklist, sent_ins_ids, sent_ins_mask, sent_cand_list


if __name__ == '__main__':
    claim_2_all_sents_of_top5_articles('Dan Roth works in University of Pennsylvania')
