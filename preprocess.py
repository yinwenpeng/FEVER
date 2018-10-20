import codecs
import ast
import os
import nltk
from common_functions import get_continuous_chunks,F1_two_strset, recall_first_set
import heapq
import json
import jsonlines
from collections import defaultdict


root = '/save/wenpeng/datasets/FEVER/'

def denoise_raw_wiki_sentlist(sentlist):
    # print 'sentlist:', sentlist
    newlist = []
    for sent in sentlist:
        if len(sent)>0:
            parts = sent.split('\t')
            if not int(parts[0].isdigit()):
                continue
            else:
                if len(parts) ==1:
                    new_sent = ''
                else:
                    new_sent = parts[1]
                newlist.append(new_sent)  # new sent can be empty

    return newlist
def reformat():
    '''
    reformat train and dev/test into:
    label, statement, #50_sent_cand, #50_binary_vec
    '''
    #wiki into {wiki_title: sent_list}
    writefile = codecs.open(root+'wiki_title2sentlist.txt', 'w', 'utf-8')
    # title2sentlist = {}
    wikipath = root+'wiki-pages/'
    page_size = 0
    for doc in os.listdir(wikipath):
        doc_path = os.path.join(wikipath, doc)
        print doc_path
        page_index = int(doc_path[-9:-6])
        # if page_index < 24:
        #     continue

        readfile = codecs.open(doc_path ,'r', 'utf-8')
        for line in readfile:
            # print 'page:', line
            line2dict = ast.literal_eval(line.strip())
            wiki_title = line2dict.get('id')
            wiki_content = line2dict.get('lines')
            if len(wiki_title) == 0 or len(wiki_content) ==0:
                continue
            raw_sent_list = wiki_content.split('\n')
            # print 'raw_sent_list:', raw_sent_list
            sent_list = denoise_raw_wiki_sentlist(raw_sent_list)
            # title2sentlist[wiki_title] = sent_list
            writefile.write(wiki_title+'\t'+'\t'.join(sent_list)+'\n')
            page_size+=1
            if page_size % 5000 == 0:
                print '...', page_size
        readfile.close()
    writefile.close()

    print 'wiki page size:', page_size


def generate_train():
    '''
    label, statement, #50sentcand, #50binaryvec
    '''
    #load wiki
    title2sentlist={}
    readwiki = codecs.open(root+'wiki_title2sentlist.txt' ,'r', 'utf-8')
    wiki_co = 0
    for line in readwiki:
        parts = line.strip().split('\t')
        title2sentlist[parts[0]] = parts[1:]
        wiki_co+=1
        if wiki_co % 1000 ==0:
            print 'wiki_co....', wiki_co
    readwiki.close()
    print 'wiki pages loaded over, totally ', len(title2sentlist), ' pages'


    label2id = {'SUPPORTS':1, 'REFUTES':0, 'NOT ENOUGH INFO':2}
    readfile = codecs.open(root+'train.jsonl' ,'r', 'utf-8')
    writefile = codecs.open(root+'train.reformat.2classes.support.refute.json' ,'w', 'utf-8')
    sent_cand_max = 20
    co = 0
    for line in readfile:
        if line.strip().find('NOT ENOUGH INFO') <0:
            # print co, line
            line2dict = ast.literal_eval(line.strip())
            labelstr = line2dict.get('label')
            labelid = label2id.get(labelstr)
            claim =  ' '.join(nltk.word_tokenize(line2dict.get('claim')))
            all_evi_list = line2dict.get('all_evidence')
            # co+=1
            # if co % 100 == 0:
            #     print co

            title2evi_idlist = {}
            for evi in all_evi_list:
                title = evi[2]
                sent_id = int(evi[3])
                evi_idlist = title2evi_idlist.get(title)
                if evi_idlist is None:
                    evi_idlist = [sent_id]
                else:
                    evi_idlist.append(sent_id)
                title2evi_idlist[title] = evi_idlist

            pos_sents = set()
            neg_sents = set()
            for title, idlist in title2evi_idlist.iteritems():

                id_set = set(idlist)

                title_sents = title2sentlist.get(title)
                if title_sents is None:
                    # print 'title_sents is None:', title
                    continue
                else:
                    for idd, sent_str  in enumerate(title_sents):
                        if len(sent_str) > 0:
                            if idd in id_set:
                                pos_sents.add(sent_str)
                            else:
                                neg_sents.add(sent_str)
            #build
            # print 'pos_sents:', pos_sents, len(pos_sents)
            # print 'neg_sents:', neg_sents, len(neg_sents)
            pos_sent_list = list(pos_sents)
            evi_size = len(pos_sents)
            if evi_size == 0:
                continue
            else:
                if len(neg_sents) > 0:
                    if evi_size > sent_cand_max:
                        pos_sent_list = pos_sent_list[:sent_cand_max]
                        neg_sent_list = []
                    else:
                        rand_sample_neg_sent_size = sent_cand_max - len(pos_sent_list)
                        # print 'rand_sample_neg_sent_size:', rand_sample_neg_sent_size
                        if rand_sample_neg_sent_size > 0:
                            if rand_sample_neg_sent_size <= len(neg_sents):
                                neg_sent_list = list(neg_sents)[:rand_sample_neg_sent_size]
                            else:
                                append_size = rand_sample_neg_sent_size - len(neg_sents)
                                neg_sent_list = list(neg_sents)
                                neg_sent_list_lastOne = neg_sent_list[-1:]
                                neg_sent_list = neg_sent_list + neg_sent_list_lastOne*append_size
                        else:
                            neg_sent_list = []
                else:
                    pos_sent_list_lastOne = pos_sent_list[-1:]
                    pos_sent_list = (pos_sent_list+pos_sent_list_lastOne*sent_cand_max)[:sent_cand_max]
                    neg_sent_list = []

                ind_list = [1]*len(pos_sent_list) + [0]*len(neg_sent_list)
                if len(ind_list) != sent_cand_max:
                    print 'len(ind_list) != sent_cand_max:', ind_list, len(ind_list), sent_cand_max
                    exit(0)

                all_sent_cand = pos_sent_list + neg_sent_list


                instance_dict = {}
                instance_dict['label'] = labelid
                instance_dict['claim'] = claim
                instance_dict['sent_cands'] = all_sent_cand
                instance_dict['sent_binary'] = ind_list
                json.dump(instance_dict, writefile)
                writefile.write('\n')

                # raw_sent_names = set([evi[0]+'\t'+str(evi[1]) for evi in sent_name_list])
                # raw_ground = set([evi[2]+'\t'+str(evi[3]) for evi in all_evi_list])
                # ground_list = [[evi.split('\t')[0],int(evi.split('\t')[1])] for evi in raw_ground]
                # instance_dict['ground_truth_names'] = ground_list
                # json.dump(instance_dict, writefile)
                # writefile.write('\n')

                # writefile.write(str(labelid)+'\t'+claim+'\t'+'\t'.join(all_sent_cand)+'\t'+' '.join(map(str,ind_list))+'\n')
                co+=1
                if co % 100 == 0:
                    print co
    writefile.close()
    readfile.close()
    print 'reformat train.jsonl over'

def doc_ranker(claim, title2wordlist):
    '''
    topN_docIDs_given_claim
    1, if entity-title perfect match, use the title
    2, if no perfect match, compare i) match(entity, title); ii) overlap(claim_vocab, page_vocab)
    '''

    entity_list, claim_wordlist = get_continuous_chunks(claim)
    claim_vocab = set(claim_wordlist)
    title2score = {}
    for title, wordlist in title2wordlist.iteritems():
        title_vocab = set(title.replace('-LRB-','').replace('-RRB-','').split('_'))
        if len(title_vocab & claim_vocab) ==0:
            continue
        else:
            #[0.0] is put in case entity_list is empty
            '''
            1, title must by subsequence, instead of recall
            '''
            score_by_title =max([recall_first_set(title_vocab,claim_vocab)]+[0.0]+[recall_first_set(title_vocab, set(entity.split())) for entity in entity_list] )
            if score_by_title == 0.0:
                continue
            elif score_by_title == 1.0:
                title2score[title] = score_by_title
                continue
            else:
                score_by_vocab = recall_first_set(claim_vocab, set(wordlist))
                title2score[title] = score_by_title+score_by_vocab
    return heapq.nlargest(100, title2score, key=title2score.get), entity_list

def doc_ranker_simple(claim, word2titlelist, title2wordlist):
    '''
    topN_docIDs_given_claim
    1, if entity-title perfect match, use the title
    2, if no perfect match, compare i) match(entity, title); ii) overlap(claim_vocab, page_vocab)
    '''
    if claim[-1] == '.':
        claim = claim[:-1]
    claim_vocab = set(claim.split())
    title2score = {}
    used_titles = set()
    for word in claim_vocab:
        title_subset = word2titlelist.get(word)
        if title_subset is not None:
            for title in title_subset:
                if title not in used_titles:
                    page_vocab = set(title2wordlist.get(title))
                    score_by_vocab = recall_first_set(claim_vocab, page_vocab)
                    if score_by_vocab > 0.5:
                        title2score[title] = score_by_vocab
                    used_titles.add(title)
    return heapq.nlargest(5, title2score, key=title2score.get)

def statistic_dev_eval():
    '''
    counting the size of each size of ground wiki pages
    {1: 0.8366925987883906, 2: 0.12348345812151824, 3: 0.018482727257962354, 4: 0.007406084033067515, 5: 0.004580078283607542, 6: 0.0028503678679898, 7: 0.002135745724448198, 8: 0.0014779685241428594, 9: 0.0010069675658995307, 10: 0.0005765701385392473, 11: 0.001307433694434068})
    '''
    size2co=defaultdict(int)
    # filelist = ['train.jsonl', 'paper_test.jsonl', 'paper_dev.jsonl']
    filelist = ['paper_test.jsonl']
    for fil in filelist:
        readfile = jsonlines.open(root+fil ,'r') #paper_test.jsonl
        size = 0
        for line2dict in readfile:
            if line2dict.get('label') != 'NOT ENOUGH INFO':
                all_evi_list = line2dict.get('evidence')
                gold_doc_list = []
                for tup in all_evi_list:
                    for i in range(len(tup)):
                        gold_doc_list.append(tup[i][2])
                doc_size = len(set(gold_doc_list))
                if doc_size > 10:
                    doc_size = 11
                size2co[doc_size] +=1
        readfile.close()
    print 'statistic over:', size2co
    all_co = sum(size2co.values())
    for size in size2co:
        size2co[size] = size2co.get(size)*1.0/all_co
    print size2co
    '''
    counting the size of evidence sentence
    '''
    size2co=defaultdict(int)
    # filelist = ['train.jsonl', 'paper_test.jsonl', 'paper_dev.jsonl']
    filelist = [ 'paper_test.jsonl']
    for fil in filelist:
        readfile = jsonlines.open(root+fil ,'r') #paper_test.jsonl
        size = 0
        for line2dict in readfile:
            if line2dict.get('label') != 'NOT ENOUGH INFO':
                all_evi_list = line2dict.get('evidence')
                # gold_doc_list = [tup[0][2] for tup in all_evi_list]
                doc_size = len(all_evi_list)
                if doc_size > 10:
                    doc_size = 11
                size2co[doc_size] +=1
        readfile.close()
    print 'statistic over:', size2co
    all_co = sum(size2co.values())
    for size in size2co:
        size2co[size] = size2co.get(size)*1.0/all_co
    print size2co

def count_sent_page(all_evi_list):
    #all_evi_list: [[title, sent_index]]
    gold_doc_list = []
    for tup in all_evi_list:
        gold_doc_list.append(tup[0])
    doc_size = len(set(gold_doc_list))
    if doc_size > 3:  #1,2,3 > 3
        doc_size = 4

    sent_size = len(all_evi_list)
    if sent_size > 4: #1,2,3,4, > 4
        sent_size = 5
    return sent_size, doc_size
def generate_dev_eval_doc_ranker():
    #rank wiki pages for each claim
    #load wiki
    # title2sentlist={}
    title2wordlist = {}
    readwiki = codecs.open(root+'wiki_title2sentlist.txt' ,'r', 'utf-8')
    wiki_co = 0
    for line in readwiki:
        parts = line.strip().split('\t')
        # title2sentlist[parts[0]] = parts[1:]
        title2wordlist[parts[0]] = parts[0].replace('-LRB-','').replace('-RRB-','').split('_')+line.strip().split()
        wiki_co+=1
        if wiki_co % 1000000 ==0:
            print 'wiki_co....', wiki_co
        # if wiki_co == 100000:
        #     break
    readwiki.close()
    print 'wiki pages loaded over, totally ', len(title2wordlist), ' pages'


    label2id = {'SUPPORTS':1, 'REFUTES':0, 'NOT ENOUGH INFO':2}
    # readfile = codecs.open(root+'shared_task_dev.jsonl' ,'r', 'utf-8')
    readfile = jsonlines.open(root+'paper_dev.jsonl' ,'r')
    # writefile = codecs.open(root+'shared_task_dev.jsonl.top5.ranked.wikipages.2classes.txt' ,'w', 'utf-8') #id, doc_id_list
    writefile = codecs.open(root+'paper_dev.jsonl.top100.ranked.wikipages.2classes.txt' ,'w', 'utf-8')
    sent_cand_max = 20
    co = 0
    doc_recall_at_1 = 0.0
    doc_recall_at_5 = 0.0
    doc_recall_at_10 = 0.0
    doc_recall_at_25 = 0.0
    doc_recall_at_50 = 0.0
    doc_recall_at_100 = 0.0
    size = 0
    for line2dict in readfile:
        # if line.strip().find('NOT ENOUGH INFO') <0: #'SUPPORTS or REFUTES
        if line2dict.get('label') != 'NOT ENOUGH INFO':
            # print co, line
            # line2dict = ast.literal_eval(line.strip())
            # labelstr = line2dict.get('label')
            instance_id = line2dict.get('id')
            # labelid = label2id.get(labelstr)
            claim = line2dict.get('claim')
            # claim =  ' '.join(nltk.word_tokenize(raw_claim))
            all_evi_list = line2dict.get('evidence')
            gold_doc_list = [tup[0][2] for tup in all_evi_list]

            doc_list, claim_entities = doc_ranker(claim,title2wordlist) # return top 100
            if len(doc_list)==0:
                doc_list = [' ',' ',' ',' ',' ']
            writefile.write(str(instance_id)+'\t'+claim+'\t'+'::'.join(claim_entities)+'\t'+'\t'.join(doc_list)+'\n')
            gold_doc_set = set(gold_doc_list)
            if gold_doc_set.issubset(set(doc_list)):
                doc_recall_at_100+=1.0
            if gold_doc_set.issubset(set(doc_list[:-50])):
                doc_recall_at_50+=1.0
            if gold_doc_set.issubset(set(doc_list[:-75])):
                doc_recall_at_25+=1.0
            if gold_doc_set.issubset(set(doc_list[:-90])):
                doc_recall_at_10+=1.0
            if gold_doc_set.issubset(set(doc_list[:-95])):
                doc_recall_at_5+=1.0
            if gold_doc_set.issubset(set(doc_list[:-99])):
                doc_recall_at_1+=1.0
            size+=1

            if size % 10 == 0:
                print  'size: ', size, ', recall_mean:', doc_recall_at_1/size,doc_recall_at_5/size,doc_recall_at_10/size,doc_recall_at_25/size,doc_recall_at_50/size,doc_recall_at_100/size
    writefile.close()
    readfile.close()

def generate_dev():
    '''
    label, statement, #all_sent_cand, #name_sent_cand, #ground_truth_sent
    '''
    #load wiki
    title2sentlist={}
    readwiki = codecs.open(root+'wiki_title2sentlist.txt' ,'r', 'utf-8')
    wiki_co = 0
    for line in readwiki:
        parts = line.strip().split('\t')
        title2sentlist[parts[0]] = parts[1:]
        wiki_co+=1
        if wiki_co % 1000000 ==0:
            print 'wiki_co....', wiki_co
    readwiki.close()
    print 'wiki pages loaded over, totally ', len(title2sentlist), ' pages'

    dev_id2docs = {}
    readfile = codecs.open(root+'shared_task_dev.jsonl.top5.ranked.wikipages.2classes.txt' ,'r', 'utf-8') #id, doc_id_list
    for line in readfile:
        parts = line.strip().split('\t')
        if len(parts) > 3:
            idd = int(parts[0])
            sentlist = parts[3:]
            dev_id2docs[idd]  = sentlist
    readfile.close()
    print 'dev ranked docs loaded over, totally ', len(dev_id2docs), ' valid instances'




    label2id = {'SUPPORTS':1, 'REFUTES':0, 'NOT ENOUGH INFO':2}
    # readfile = codecs.open(root+'shared_task_dev.jsonl' ,'r', 'utf-8')
    readfile = jsonlines.open(root+'shared_task_dev.jsonl' ,'r')
    writefile = codecs.open(root+'dev.jsonl.reformat.2classes.support.refute.json' ,'w', 'utf-8')
    sent_cand_max = 20
    co = 0
    full_cover = 0
    for line in readfile:
        if line.get('label') != 'NOT ENOUGH INFO':
            # print co, line
            # line2dict = ast.literal_eval(line.strip())
            labelstr = line.get('label')
            idd = line.get('id')
            labelid = label2id.get(labelstr)
            instance_id = line.get('id')
            raw_claim = line.get('claim')
            tokenized_claim =  ' '.join(nltk.word_tokenize(raw_claim))
            all_evi_list = line.get('all_evidence')



            sent_cand_list = []
            sent_name_list = []
            doc_cands = dev_id2docs.get(instance_id)
            if doc_cands is not None:
                for doc in doc_cands:
                    doc2sents = title2sentlist.get(doc)
                    if doc2sents is not None:
                        for i, sent in enumerate(doc2sents):
                            if len(sent.strip()) == 0:
                                continue
                            else:
                                sent_cand_list.append(sent)
                                sent_name = [doc, i]
                                sent_name_list.append(sent_name)
            if len(sent_cand_list) == 0:
                continue
            else:
                instance_dict = {}
                instance_dict['label'] = labelid
                instance_dict['id'] = idd
                instance_dict['claim'] = tokenized_claim
                instance_dict['sent_cands'] = sent_cand_list
                instance_dict['sent_names'] = sent_name_list
                raw_sent_names = set([evi[0]+'\t'+str(evi[1]) for evi in sent_name_list])
                raw_ground = set([evi[2]+'\t'+str(evi[3]) for evi in all_evi_list])
                ground_list = [[evi.split('\t')[0],int(evi.split('\t')[1])] for evi in raw_ground]
                instance_dict['ground_truth_names'] = ground_list
                json.dump(instance_dict, writefile)
                writefile.write('\n')

                if raw_ground.issubset(raw_sent_names):
                    full_cover+=1
                co+=1

    writefile.close()
    '''
    dev json write over, full cover rato: 0.803405340534
    '''
    print 'dev json write over, full cover rato:', full_cover*1.0/co

def compute_f1_two_list_names(pred_names, gold_names):
    # print 'pred_names: ', pred_names
    # print 'gold_names: ',gold_names
    pred_names = [lis[0]+'-'+str(lis[1])  for lis in pred_names]
    gold_names = [lis[0]+'-'+str(lis[1])  for lis in gold_names]

    pred_set = set(pred_names)
    gold_set = set(gold_names)
    pred_size = len(pred_set)
    gold_size = len(gold_set)

    overlap_size = len(pred_set&gold_set)
    if overlap_size == 0:
        return 0.0, 0.0, 0.0
    recall = overlap_size*1.0/gold_size
    precision = overlap_size*1.0/pred_size
    return 2.0*recall*precision/(1e-8+recall+precision), recall, precision

def compute_f1_recall_two_list_names(pred_names, gold_names):
    pred_set = set(pred_names)
    gold_set = set(gold_names)
    pred_size = len(pred_set)
    gold_size = len(gold_set)

    overlap_size = len(pred_set&gold_set)
    if overlap_size == 0:
        return 0.0, 0.0
    recall = overlap_size*1.0/gold_size
    precision = overlap_size*1.0/pred_size
    return 2.0*recall*precision/(1e-8+recall+precision), recall

def generate_dev_eval_doc_ranker_3th_class():
    #rank wiki pages for each claim
    #load wiki
    title2wordlist = {}
    word2titlelist=defaultdict(list)
    readwiki = codecs.open(root+'wiki_title2sentlist.txt' ,'r', 'utf-8')
    wiki_co = 0
    for line in readwiki:
        parts = line.strip().split('\t')
        title = parts[0]
        title_wordlist = title.replace('-LRB-','').replace('-RRB-','').split('_')
        title2wordlist[title] = title_wordlist+line.strip().split()

        title_vocab = set(title_wordlist)
        for word in title_vocab:
            word2titlelist[word].append(title)
        wiki_co+=1
        if wiki_co % 1000000 ==0:
            print 'wiki_co....', wiki_co
    readwiki.close()
    print 'wiki pages loaded over, totally ', len(title2wordlist), ' pages'


    # label2id = {'SUPPORTS':1, 'REFUTES':0, 'NOT ENOUGH INFO':2}
    readfile = jsonlines.open(root+'shared_task_dev.jsonl' ,'r')
    writefile = codecs.open(root+'shared_task_dev.jsonl.top5.simple-ranked.wikipages.NoEnoughInfo.txt' ,'w', 'utf-8') #id, doc_id_list
    size = 0
    for line in readfile:
        if line.get('label') == 'NOT ENOUGH INFO': #'SUPPORTS or REFUTES
            instance_id = line.get('id')
            claim = line.get('claim')
            # doc_list, claim_entities = doc_ranker(claim,title2wordlist)
            doc_list = doc_ranker_simple(claim,word2titlelist, title2wordlist)
            if len(doc_list)==0:
                doc_list = [' ',' ',' ',' ',' ']
            writefile.write(str(instance_id)+'\t'+claim+'\t'+'\t'.join(doc_list)+'\n')
            size+=1

            if size % 10 == 0:
                print size
    writefile.close()
    readfile.close()


def generate_train_eval_doc_ranker_3th_class():
    #rank wiki pages for each claim
    #load wiki
    title2wordlist = {}
    word2titlelist=defaultdict(list)
    readwiki = codecs.open(root+'wiki_title2sentlist.txt' ,'r', 'utf-8')
    wiki_co = 0
    for line in readwiki:
        parts = line.strip().split('\t')
        title = parts[0]
        title_wordlist = title.replace('-LRB-','').replace('-RRB-','').split('_')
        title2wordlist[title] = title_wordlist+line.strip().split()

        title_vocab = set(title_wordlist)
        for word in title_vocab:
            word2titlelist[word].append(title)
        wiki_co+=1
        if wiki_co % 1000000 ==0:
            print 'wiki_co....', wiki_co
    readwiki.close()
    print 'wiki pages loaded over, totally ', len(title2wordlist), ' pages'


    # label2id = {'SUPPORTS':1, 'REFUTES':0, 'NOT ENOUGH INFO':2}
    readfile = jsonlines.open(root+'train.jsonl' ,'r')
    writefile = codecs.open(root+'train.jsonl.top5.simple-ranked.wikipages.NoEnoughInfo.txt' ,'w', 'utf-8') #id, doc_id_list
    size = 0
    for line in readfile:
        if line.get('label') == 'NOT ENOUGH INFO': #'SUPPORTS or REFUTES
            instance_id = line.get('id')
            claim = line.get('claim')
            # doc_list, claim_entities = doc_ranker(claim,title2wordlist)
            doc_list = doc_ranker_simple(claim,word2titlelist, title2wordlist)

            if len(doc_list)==0:
                doc_list = [' ',' ',' ',' ',' ']
            writefile.write(str(instance_id)+'\t'+claim+'\t'+'\t'.join(doc_list)+'\n')
            size+=1

            if size % 10 == 0:
                print size
    writefile.close()
    readfile.close()

def generate_full_dev():
    #load wiki
    title2sentlist={}
    readwiki = codecs.open(root+'wiki_title2sentlist.txt' ,'r', 'utf-8')
    wiki_co = 0
    for line in readwiki:
        parts = line.strip().split('\t')
        title2sentlist[parts[0]] = parts[1:]
        wiki_co+=1
        if wiki_co % 1000000 ==0:
            print 'wiki_co....', wiki_co
    readwiki.close()
    print 'wiki pages loaded over, totally ', len(title2sentlist), ' pages'
    #first load dev 2 2classes

    readfile = codecs.open(root+'shared_task_dev.jsonl.top5.simple-ranked.wikipages.NoEnoughInfo.txt' ,'r', 'utf-8')
    writefile = codecs.open(root+'dev.jsonl.reformat.3th.class.json' ,'w', 'utf-8')
    for line in readfile:
        parts  =  line.strip().split('\t')
        if len(parts) > 2:
            raw_claim = parts[1]
            idd = int(parts[0])
            tokenized_claim =  ' '.join(nltk.word_tokenize(raw_claim))
            first_title = parts[2]
            sent_cand_list = title2sentlist.get(first_title)

            instance_dict = {}
            instance_dict['id'] = idd
            instance_dict['label'] = 2
            instance_dict['claim'] = tokenized_claim
            instance_dict['sent_cands'] = sent_cand_list

            json.dump(instance_dict, writefile)
            writefile.write('\n')
    writefile.close()
    readfile.close()

def generate_full_train():
    #load wiki
    title2sentlist={}
    readwiki = codecs.open(root+'wiki_title2sentlist.txt' ,'r', 'utf-8')
    wiki_co = 0
    for line in readwiki:
        parts = line.strip().split('\t')
        title2sentlist[parts[0]] = parts[1:]
        wiki_co+=1
        if wiki_co % 1000000 ==0:
            print 'wiki_co....', wiki_co
    readwiki.close()
    print 'wiki pages loaded over, totally ', len(title2sentlist), ' pages'
    #first load dev 2 2classes

    readfile = codecs.open(root+'train.jsonl.top5.simple-ranked.wikipages.NoEnoughInfo.txt' ,'r', 'utf-8')
    writefile = codecs.open(root+'train.reformat.3th.class.json' ,'w', 'utf-8')
    for line in readfile:
        parts  =  line.strip().split('\t')
        if len(parts) > 2:
            raw_claim = parts[1]
            tokenized_claim =  ' '.join(nltk.word_tokenize(raw_claim))
            first_title = parts[2]
            sent_cand_list = title2sentlist.get(first_title)

            instance_dict = {}
            instance_dict['label'] = 2
            instance_dict['claim'] = tokenized_claim
            instance_dict['sent_cands'] = sent_cand_list

            json.dump(instance_dict, writefile)
            writefile.write('\n')
    writefile.close()
    readfile.close()

def split_sharedDev_paperDevTest():
    # load paper dev
    readfile = jsonlines.open(root+'paper_dev.jsonl' ,'r')
    size = 0
    dev_ids = set()
    for line in readfile:

        idd = line.get('id')
        dev_ids.add(idd)
        size+=1

        # if size % 10 == 0:
        #     print size
    readfile.close()
    readfile = jsonlines.open(root+'paper_test.jsonl' ,'r')
    # writedev = codecs.open(root+'train.jsonl.top5.simple-ranked.wikipages.NoEnoughInfo.txt' ,'w', 'utf-8') #id, doc_id_list
    size = 0
    test_ids = set()
    for line in readfile:

        idd = line.get('id')
        test_ids.add(idd)
        size+=1

        # if size % 10 == 0:
        #     print size
    readfile.close()
    print 'dev ids and test ids split over, size: ', len(dev_ids), len(test_ids)
    #write into file
    writedev = codecs.open(root+'paper_dev.jsonl.reformat.2classes.support.refute.json' ,'w', 'utf-8') #id, doc_id_list
    writetest = codecs.open(root+'paper_test.jsonl.reformat.2classes.support.refute.json' ,'w', 'utf-8') #id, doc_id_list
    readfile =    codecs.open(root+'dev.jsonl.reformat.2classes.support.refute.json' ,'r', 'utf-8')
    for line in readfile:
        line2dict = json.loads(line)
        idd  = line2dict.get('id')
        if idd in dev_ids:
            json.dump(line2dict, writedev)
            writedev.write('\n')
        else:
            json.dump(line2dict, writetest)
            writetest.write('\n')
    readfile.close()
    writedev.close()
    writetest.close()
    print 'dev 2classes split over'

    writedev = codecs.open(root+'paper_dev.jsonl.reformat.3th.class.json' ,'w', 'utf-8') #id, doc_id_list
    writetest = codecs.open(root+'paper_test.jsonl.reformat.3th.class.json' ,'w', 'utf-8') #id, doc_id_list
    readfile =    codecs.open(root+'dev.jsonl.reformat.3th.class.json' ,'r', 'utf-8')
    for line in readfile:
        line2dict = json.loads(line)
        idd  = line2dict.get('id')
        if idd in dev_ids:
            json.dump(line2dict, writedev)
            writedev.write('\n')
        else:
            json.dump(line2dict, writetest)
            writetest.write('\n')
    readfile.close()
    writedev.close()
    writetest.close()


if __name__ == '__main__':
    '''
    generate train and dev for 2 classes
    '''
    # reformat()
    # generate_train()
    generate_dev_eval_doc_ranker()
    # generate_dev()
    '''
    doc rank for NoEngouhInfo in both dev and train
    '''
    # generate_dev_eval_doc_ranker_3th_class()
    # generate_train_eval_doc_ranker_3th_class()

    '''
    generate train and dev for full 3 classes
    '''
    # generate_full_dev()
    # generate_full_train()
    '''
    split shared dev into true dev and test
    '''
    # split_sharedDev_paperDevTest()
    '''
    statistics
    '''
    # statistic_dev_eval()
