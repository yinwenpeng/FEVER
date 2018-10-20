import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time
import math
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import random
import json
import codecs
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from theano.tensor.signal import downsample
from random import shuffle
from theano.tensor.nnet.bn import batch_normalization

from load_data import load_fever_train, load_RTE_dataset_as_test, load_fever_dev, load_fever_test,load_fever_train_NoEnoughInfo, load_fever_dev_NoEnoughInfo, load_fever_test_NoEnoughInfo,load_SciTailV1_dataset
from common_functions import store_model_to_file,Attentive_Conv_for_Pair_easy_version,load_model_from_file,load_word2vec,load_word2vec_to_init, store_model_to_file, elementwise_is_two,Conv_with_Mask_with_Gate, Conv_with_Mask, create_conv_para, L2norm_paraList, create_HiddenLayer_para, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para

from preprocess import compute_f1_two_list_names,compute_f1_recall_two_list_names
from fever_scorer import fever_score
'''
0.515384615385 on RTE-5; 0.544651162791 on SciTail
'''

def evaluate_lenet5(learning_rate=0.02, n_epochs=100, emb_size=300, batch_size=50, filter_size=[3], sent_len=40, claim_len=40, cand_size=10,hidden_size=[300,300], max_pred_pick=5):

    model_options = locals().copy()
    print "model options", model_options

    pred_id2label = {1:'SUPPORTS', 0:'REFUTES', 2:'NOT ENOUGH INFO'}
    root = '/save/wenpeng/datasets/FEVER/'
    seed=1234
    np.random.seed(seed)
    rng = np.random.RandomState(seed)    #random seed, control the model generates the same results
    srng = T.shared_randomstreams.RandomStreams(rng.randint(seed))

    "load raw data"
    vocabfile = codecs.open(root+'word2id.txt', 'r', 'utf-8')
    word2id= json.loads(vocabfile.read())
    # co=0
    # for line in vocabfile:
    #     word2id = json.loads(line)
    #     co+=1
    # print 'co: ', co
    # word2id = json.load(open(root+'word2id.json')) #json.loads(vocabfile)
    vocabfile.close()
    print 'load word2id over'
    # train_sents, train_sent_masks, train_sent_labels, train_claims, train_claim_mask, train_labels, word2id  = load_fever_train(sent_len, claim_len, cand_size)
    # train_3th_sents, train_3th_sent_masks, train_3th_sent_labels, train_3th_claims, train_3th_claim_mask, train_3th_labels, word2id = load_fever_train_NoEnoughInfo(sent_len, claim_len, cand_size, word2id)
    all_sentences_l, all_masks_l, all_sentences_r, all_masks_r, all_labels, _ = load_SciTailV1_dataset(sent_len, word2id)
    # all_sentences_l, all_masks_l, all_sentences_r, all_masks_r, all_labels, _ = load_RTE_dataset_as_test(sent_len, word2id)





    # dev_sents_l=np.asarray(all_sentences_l[1], dtype='int32')
    test_sents_l=np.asarray(all_sentences_l[2], dtype='int32')

    # dev_masks_l=np.asarray(all_masks_l[1], dtype=theano.config.floatX)
    test_masks_l=np.asarray(all_masks_l[2], dtype=theano.config.floatX)

    # dev_sents_r=np.asarray(all_sentences_r[1]    , dtype='int32')
    test_sents_r=np.asarray(all_sentences_r[2]    , dtype='int32')

    # dev_masks_r=np.asarray(all_masks_r[1], dtype=theano.config.floatX)
    test_masks_r=np.asarray(all_masks_r[2], dtype=theano.config.floatX)

    # dev_labels_store=np.asarray(all_labels[1], dtype='int32')
    test_labels_store=np.asarray(all_labels[2], dtype='int32')

    # dev_size=len(dev_labels_store)
    test_size=len(test_labels_store)


    vocab_size=len(word2id)+1

    print 'vocab size: ', vocab_size



    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    # id2word = {y:x for x,y in word2id.iteritems()}
    # word2vec=load_word2vec()
    # rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    init_embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    "now, start to build the input form of the model"
    sents_ids=T.imatrix() #(batch, cand_size, sent_len)
    sents_mask=T.fmatrix()
    # sents_labels=T.imatrix() #(batch, cand_size)
    # claim_ids = T.imatrix() #(batch, claim_len)
    # claim_mask = T.fmatrix()

    # joint_sents_ids=T.itensor3() #(batch, cand_size, sent_len)
    # joint_sents_mask=T.ftensor3()
    # joint_sents_labels=T.imatrix() #(batch, cand_size)
    claim_ids = T.imatrix() #(batch, claim_len)
    claim_mask = T.fmatrix()
    labels=T.ivector()

    # test_premise_ids = T.imatrix()
    # test_premise_matrix = T.fmatrix()
    # test_hypo_ids = T.imatrix()
    # test_hypo_matrix = T.fmatrix()
    # test_scitail_minibatch_labels = T.ivector()

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'


    embed_input_sents=init_embeddings[sents_ids.flatten()].reshape((batch_size, sent_len, emb_size)).dimshuffle(0,2,1)#embed_input(init_embeddings, sents_ids_l)#embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    embed_input_claim=init_embeddings[claim_ids.flatten()].reshape((batch_size,sent_len, emb_size)).dimshuffle(0,2,1)



    conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    # task1_att_conv_W, task1_att_conv_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    # task1_conv_W_context, task1_conv_b_context=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, 1))
    att_conv_W, att_conv_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    conv_W_context, conv_b_context=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, 1))

    NN_para=[conv_W, conv_b, att_conv_W, att_conv_b,conv_W_context,conv_b_context]


    '''
    training task2, predict 3 labels
    '''
    joint_embed_input_sents=init_embeddings[sents_ids.flatten()].reshape((batch_size, sent_len, emb_size)).dimshuffle(0,2,1)#embed_input(init_embeddings, sents_ids_l)#embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    joint_embed_input_claim=init_embeddings[claim_ids.flatten()].reshape((batch_size,sent_len, emb_size)).dimshuffle(0,2,1)
    joint_conv_model_sents = Conv_with_Mask(rng, input_tensor3=joint_embed_input_sents,
             mask_matrix = sents_mask,
             image_shape=(batch_size, 1, emb_size, sent_len),
             filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]), W=conv_W, b=conv_b)    #mutiple mask with the conv_out to set the features by UNK to zero
    joint_premise_emb=joint_conv_model_sents.maxpool_vec #(batch_size*cand_size, hidden_size) # each sentence then have an embedding of length hidden_size
    # joint_batch_sent_emb = joint_sent_embeddings.reshape((batch_size, cand_size, hidden_size[0]))
    # joint_premise_emb = T.sum(joint_batch_sent_emb*joint_sents_labels.dimshuffle(0,1,'x'), axis=1) #(batch, hidden_size)

    joint_conv_model_claims = Conv_with_Mask(rng, input_tensor3=joint_embed_input_claim,
             mask_matrix = claim_mask,
             image_shape=(batch_size, 1, emb_size, claim_len),
             filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]), W=conv_W, b=conv_b)    #mutiple mask with the conv_out to set the features by UNK to zero
    joint_claim_embeddings=joint_conv_model_claims.maxpool_vec #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

    joint_premise_hypo_emb = T.concatenate([joint_premise_emb,joint_claim_embeddings], axis=1) #(batch, 2*hidden_size)
    '''
    attentive conv in task2
    '''
    # joint_sents_tensor3 = joint_embed_input_sents.dimshuffle(0,2,1).reshape((batch_size, cand_size*sent_len, emb_size))
    # joint_sents_dot = T.batched_dot(joint_sents_tensor3, joint_sents_tensor3.dimshuffle(0,2,1)) #(batch_size, cand_size*sent_len, cand_size*sent_len)
    # joint_sents_dot_2_matrix = T.nnet.softmax(joint_sents_dot.reshape((batch_size*cand_size*sent_len, cand_size*sent_len)))
    # joint_sents_context = T.batched_dot(joint_sents_dot_2_matrix.reshape((batch_size, cand_size*sent_len, cand_size*sent_len)), joint_sents_tensor3) #(batch_size, cand_size*sent_len, emb_size)
    # joint_add_sents_context = joint_embed_input_sents+joint_sents_context.reshape((batch_size*cand_size, sent_len, emb_size)).dimshuffle(0,2,1)#T.concatenate([joint_embed_input_sents, joint_sents_context.reshape((batch_size*cand_size, sent_len, emb_size)).dimshuffle(0,2,1)], axis=1) #(batch_size*cand_size, 2*emb_size, sent_len)

    attentive_conv_layer = Attentive_Conv_for_Pair_easy_version(rng,
            input_tensor3=joint_embed_input_sents, #batch_size*cand_size, 2*emb_size, sent_len
            input_tensor3_r = joint_embed_input_claim,
             mask_matrix = sents_mask,
             mask_matrix_r = claim_mask,
             image_shape=(batch_size, 1, emb_size, sent_len),
             image_shape_r = (batch_size, 1, emb_size, sent_len),
             filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]),
             filter_shape_context=(hidden_size[0], 1,emb_size, 1),
             W=att_conv_W, b=att_conv_b,
             W_context=conv_W_context, b_context=conv_b_context)
    attentive_sent_embeddings_l = attentive_conv_layer.attentive_maxpool_vec_l #(batch_size*cand_size, hidden_size)
    attentive_sent_embeddings_r = attentive_conv_layer.attentive_maxpool_vec_r
    "Logistic Regression layer"
    joint_LR_input = T.concatenate([joint_premise_hypo_emb,attentive_sent_embeddings_l,attentive_sent_embeddings_r], axis=1)
    joint_LR_input_size=2*hidden_size[0]+2*hidden_size[0]

    joint_U_a = create_ensemble_para(rng, 3, joint_LR_input_size) # (input_size, 3)
    joint_LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    joint_LR_para=[joint_U_a, joint_LR_b]

    joint_layer_LR=LogisticRegression(rng, input=joint_LR_input, n_in=joint_LR_input_size, n_out=3, W=joint_U_a, b=joint_LR_b) #basically it is a multiplication between weight matrix and input feature vector
    # joint_loss=joint_layer_LR.negative_log_likelihood(joint_labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.







    '''
    testing
    joint_sents_ids=T.itensor3() #(batch, cand_size, sent_len)
    joint_sents_mask=T.ftensor3()
    joint_sents_labels=T.imatrix() #(batch, cand_size)
    joint_claim_ids = T.imatrix() #(batch, claim_len)
    joint_claim_mask = T.fmatrix()
    joint_labels=T.ivector()
    '''
    pred_minibatch_labels = joint_layer_LR.y_pred
    pred_minibatch_labels_2_2classes =  T.where( pred_minibatch_labels > 1, 0, pred_minibatch_labels)

    pred_minibatch_error = T.mean(T.neq(pred_minibatch_labels_2_2classes, labels))

    params = [init_embeddings]+NN_para + joint_LR_para
    load_model_from_file(root+'para_for_test_scitail', params)

    # train_model = theano.function([sents_ids,sents_mask,sents_labels,claim_ids,claim_mask,joint_sents_ids,joint_sents_mask,joint_sents_labels, joint_claim_ids, joint_claim_mask, joint_labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([sents_ids,sents_mask,claim_ids, claim_mask, labels], pred_minibatch_error, allow_input_downcast=True, on_unused_input='ignore')
    # dev_model = theano.function([joint_sents_ids,joint_sents_mask,joint_sents_labels, joint_claim_ids, joint_claim_mask, joint_labels], pred_minibatch_error, allow_input_downcast=True, on_unused_input='ignore')

    # test_model = theano.function([sents_ids,sents_mask,sents_labels, claim_ids,claim_mask, joint_labels], [inter_matrix,test_layer_LR.errors(joint_labels), test_layer_LR.y_pred], allow_input_downcast=True, on_unused_input='ignore')
    # dev_model = theano.function([sents_ids,sents_mask,sents_labels, claim_ids,claim_mask, joint_labels], [inter_matrix,test_layer_LR.errors(joint_labels), test_layer_LR.y_pred], allow_input_downcast=True, on_unused_input='ignore')

    ###############
    # TRAIN MODEL #
    ###############
    print '... testing'
    # early-stopping parameters
    patience = 50000000000  # look as this many examples regardless
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False

    # joint_n_train_batches=joint_train_size/batch_size
    # joint_train_batch_start=list(np.arange(joint_n_train_batches)*batch_size)+[joint_train_size-batch_size]
    # n_train_batches=train_size/batch_size
    # train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]

    # n_dev_batches=dev_size/batch_size
    # dev_batch_start=list(np.arange(n_dev_batches)*batch_size)+[dev_size-batch_size]
    n_test_batches=test_size/batch_size
    remain_test_batches=test_size%batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]


    max_acc_dev=0.0
    max_acc_test=0.0

    cost_i=0.0

    error_sum=0.0
    for idd, test_batch_id in enumerate(test_batch_start): # for each test batch
        error_i=test_model(
                test_sents_l[test_batch_id:test_batch_id+batch_size],
                test_masks_l[test_batch_id:test_batch_id+batch_size],
                test_sents_r[test_batch_id:test_batch_id+batch_size],
                test_masks_r[test_batch_id:test_batch_id+batch_size],
                test_labels_store[test_batch_id:test_batch_id+batch_size])
        error_sum+=error_i
    test_acc=1.0-error_sum/(len(test_batch_start))
    print '\tcurrent test_acc:', test_acc




if __name__ == '__main__':
    evaluate_lenet5()
