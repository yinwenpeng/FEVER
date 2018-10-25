import pickle
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
import codecs
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
# from random import shuffle

# from load_data import load_fever_train,  load_fever_dev, load_fever_test,load_fever_train_NoEnoughInfo, load_fever_dev_NoEnoughInfo, load_fever_test_NoEnoughInfo
from common_functions import store_model_to_file,Attentive_Conv_for_Pair_easy_version,load_word2vec,load_word2vec_to_init, store_model_to_file, elementwise_is_two,Conv_with_Mask_with_Gate, Conv_with_Mask, create_conv_para, L2norm_paraList, create_HiddenLayer_para, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, load_model_from_file, create_LSTM_para

# from preprocess import compute_f1_two_list_names,compute_f1_recall_two_list_names
# from fever_scorer import fever_score

from demo_preprocess import claim_input_2_theano_input
'''

'''

def evaluate_lenet5(claim, title2sentlist, title2wordlist,word2id):
    learning_rate=0.02
    n_epochs=100
    emb_size=300
    batch_size=1#50
    filter_size=[3]
    sent_len=40
    claim_len=40
    cand_size=10
    hidden_size=[300,300]
    max_pred_pick=5

    # model_options = locals().copy()
    # print("model options", model_options)
    # print('title2sentlist len', len(title2sentlist))
    # print('title2wordlist len', len(title2wordlist))

    pred_id2label = {1:'SUPPORTS', 0:'REFUTES', 2:'NOT ENOUGH INFO'}

    seed=1234
    np.random.seed(seed)
    rng = np.random.RandomState(seed)    #random seed, control the model generates the same results
    srng = T.shared_randomstreams.RandomStreams(rng.randint(seed))


    claim_idlist, claim_masklist, sent_ins_ids, sent_ins_mask, sent_cand_list = claim_input_2_theano_input(claim, word2id, claim_len, sent_len, cand_size,title2sentlist, title2wordlist)

    test_claims=np.asarray([claim_idlist], dtype='int32')
    test_claim_mask=np.asarray([claim_masklist], dtype=theano.config.floatX)

    test_sents=np.asarray([sent_ins_ids], dtype='int32')
    test_sent_masks=np.asarray([sent_ins_mask], dtype=theano.config.floatX)


    vocab_size=len(word2id)+1




    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    # id2word = {y:x for x,y in word2id.items()}
    # word2vec=load_word2vec()
    # rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    init_embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    '''
    the first block for evidence identification in two classes (support & reject)
    the second block for textual entailment: given evidence labels, predict the claim labels
    '''
    sents_ids=T.itensor3() #(batch, cand_size, sent_len)
    sents_mask=T.ftensor3()
    # sents_labels=T.imatrix() #(batch, cand_size)
    claim_ids = T.imatrix() #(batch, claim_len)
    claim_mask = T.fmatrix()

    # joint_sents_ids=T.itensor3() #(batch, cand_size, sent_len)
    # joint_sents_mask=T.ftensor3()
    # # joint_sents_labels=T.imatrix() #(batch, cand_size)
    # joint_claim_ids = T.imatrix() #(batch, claim_len)
    # joint_claim_mask = T.fmatrix()
    # joint_labels=T.ivector()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')


    embed_input_sents=init_embeddings[sents_ids.flatten()].reshape((batch_size*cand_size, sent_len, emb_size)).dimshuffle(0,2,1)#embed_input(init_embeddings, sents_ids_l)#embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    embed_input_claim=init_embeddings[claim_ids.flatten()].reshape((batch_size,claim_len, emb_size)).dimshuffle(0,2,1)


    "shared parameters"
    conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    "tasl 1 parameters"
    task1_att_conv_W, task1_att_conv_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    task1_conv_W_context, task1_conv_b_context=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, 1))
    "task 2 parameters"
    att_conv_W, att_conv_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    conv_W_context, conv_b_context=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, 1))

    NN_para=[conv_W, conv_b, task1_att_conv_W, task1_att_conv_b, att_conv_W, att_conv_b,task1_conv_W_context,conv_W_context]

    conv_model_sents = Conv_with_Mask(rng, input_tensor3=embed_input_sents,
             mask_matrix = sents_mask.reshape((sents_mask.shape[0]*sents_mask.shape[1],sents_mask.shape[2])),
             image_shape=(batch_size*cand_size, 1, emb_size, sent_len),
             filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]), W=conv_W, b=conv_b)    #mutiple mask with the conv_out to set the features by UNK to zero
    sent_embeddings=conv_model_sents.maxpool_vec #(batch_size*cand_size, hidden_size) # each sentence then have an embedding of length hidden_size
    batch_sent_emb = sent_embeddings.reshape((batch_size, cand_size, hidden_size[0]))

    conv_model_claims = Conv_with_Mask(rng, input_tensor3=embed_input_claim,
             mask_matrix = claim_mask,
             image_shape=(batch_size, 1, emb_size, claim_len),
             filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]), W=conv_W, b=conv_b)    #mutiple mask with the conv_out to set the features by UNK to zero
    claim_embeddings=conv_model_claims.maxpool_vec #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
    batch_claim_emb = T.repeat(claim_embeddings.dimshuffle(0,'x', 1), cand_size, axis=1)

    '''
    attentive conv for task1
    '''
    task1_attentive_conv_layer = Attentive_Conv_for_Pair_easy_version(rng,
            input_tensor3=embed_input_sents, #batch_size*cand_size, emb_size, sent_len
            input_tensor3_r = T.repeat(embed_input_claim, cand_size, axis=0),
             mask_matrix = sents_mask.reshape((sents_mask.shape[0]*sents_mask.shape[1],sents_mask.shape[2])),
             mask_matrix_r = T.repeat(claim_mask,cand_size, axis=0),
             image_shape=(batch_size*cand_size, 1, emb_size, sent_len),
             image_shape_r = (batch_size*cand_size, 1, emb_size, claim_len),
             filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]),
             filter_shape_context=(hidden_size[0], 1,emb_size, 1),
             W=task1_att_conv_W, b=task1_att_conv_b,
             W_context=task1_conv_W_context, b_context=task1_conv_b_context)
    task1_attentive_sent_embeddings_l = task1_attentive_conv_layer.attentive_maxpool_vec_l  #(batch_size*cand_size, hidden_size)
    task1_attentive_sent_embeddings_r = task1_attentive_conv_layer.attentive_maxpool_vec_r




    concate_claim_sent = T.concatenate([batch_claim_emb,batch_sent_emb, T.sum(batch_claim_emb*batch_sent_emb, axis=2).dimshuffle(0,1,'x')], axis=2)
    concate_2_matrix = concate_claim_sent.reshape((batch_size*cand_size, hidden_size[0]*2+1))
    "to score each evidence sentence, we use the output of attentiveConv, as well as the output of standard CNN"
    LR_input = T.concatenate([concate_2_matrix, task1_attentive_sent_embeddings_l,task1_attentive_sent_embeddings_r], axis=1)
    LR_input_size = hidden_size[0]*2+1 + hidden_size[0]*2

    # LR_input = concate_2_matrix
    # LR_input_size = hidden_size[0]*2+1
    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    U_a = create_ensemble_para(rng, 1, LR_input_size) # the weight matrix hidden_size*2
    # LR_b = theano.shared(value=np.zeros((8,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR_para=[U_a]
    # layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=8, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    score_matrix = T.nnet.sigmoid(LR_input.dot(U_a))  #batch * 12
    inter_matrix = score_matrix.reshape((batch_size, cand_size))

    # inter_sent_claim = T.batched_dot(batch_sent_emb, batch_claim_emb) #(batch_size, cand_size, 1)
    # inter_matrix = T.nnet.sigmoid(inter_sent_claim.reshape((batch_size, cand_size)))
    '''
    maybe 1.0-inter_matrix can be rewritten into 1/e^(inter_matrix)
    '''
    binarize_prob = T.where( inter_matrix > 0.5, 1, 0)  #(batch_size, cand_size)
    sents_labels = inter_matrix*binarize_prob

    '''
    training task2, predict 3 labels
    '''
    # joint_embed_input_sents=init_embeddings[joint_sents_ids.flatten()].reshape((batch_size*cand_size, sent_len, emb_size)).dimshuffle(0,2,1)#embed_input(init_embeddings, sents_ids_l)#embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    # joint_embed_input_claim=init_embeddings[joint_claim_ids.flatten()].reshape((batch_size,claim_len, emb_size)).dimshuffle(0,2,1)
    # joint_conv_model_sents = Conv_with_Mask(rng, input_tensor3=joint_embed_input_sents,
    #          mask_matrix = joint_sents_mask.reshape((joint_sents_mask.shape[0]*joint_sents_mask.shape[1],joint_sents_mask.shape[2])),
    #          image_shape=(batch_size*cand_size, 1, emb_size, sent_len),
    #          filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]), W=conv_W, b=conv_b)    #mutiple mask with the conv_out to set the features by UNK to zero
    # joint_sent_embeddings=joint_conv_model_sents.maxpool_vec #(batch_size*cand_size, hidden_size) # each sentence then have an embedding of length hidden_size
    # joint_batch_sent_emb = joint_sent_embeddings.reshape((batch_size, cand_size, hidden_size[0]))
    # "??? use joint_sents_labels means the evidence labels are not provided by task 1?"
    # joint_premise_emb = T.sum(joint_batch_sent_emb*joint_sents_labels.dimshuffle(0,1,'x'), axis=1) #(batch, hidden_size)

    premise_emb = T.sum(batch_sent_emb*sents_labels.dimshuffle(0,1,'x'), axis=1)

    # joint_conv_model_claims = Conv_with_Mask(rng, input_tensor3=joint_embed_input_claim,
    #          mask_matrix = joint_claim_mask,
    #          image_shape=(batch_size, 1, emb_size, claim_len),
    #          filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]), W=conv_W, b=conv_b)    #mutiple mask with the conv_out to set the features by UNK to zero
    # joint_claim_embeddings=joint_conv_model_claims.maxpool_vec #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

    premise_hypo_emb = T.concatenate([premise_emb,claim_embeddings], axis=1) #(batch, 2*hidden_size)
    '''
    attentive conv in task2
    '''
    sents_tensor3 = embed_input_sents.dimshuffle(0,2,1).reshape((batch_size, cand_size*sent_len, emb_size))
    sents_dot = T.batched_dot(sents_tensor3, sents_tensor3.dimshuffle(0,2,1)) #(batch_size, cand_size*sent_len, cand_size*sent_len)
    sents_dot_2_matrix = T.nnet.softmax(sents_dot.reshape((batch_size*cand_size*sent_len, cand_size*sent_len)))
    sents_context = T.batched_dot(sents_dot_2_matrix.reshape((batch_size, cand_size*sent_len, cand_size*sent_len)), sents_tensor3) #(batch_size, cand_size*sent_len, emb_size)
    add_sents_context = embed_input_sents+sents_context.reshape((batch_size*cand_size, sent_len, emb_size)).dimshuffle(0,2,1)#T.concatenate([joint_embed_input_sents, joint_sents_context.reshape((batch_size*cand_size, sent_len, emb_size)).dimshuffle(0,2,1)], axis=1) #(batch_size*cand_size, 2*emb_size, sent_len)

    attentive_conv_layer = Attentive_Conv_for_Pair_easy_version(rng,
            input_tensor3=add_sents_context, #batch_size*cand_size, 2*emb_size, sent_len
            input_tensor3_r = T.repeat(embed_input_claim, cand_size, axis=0),
             mask_matrix = sents_mask.reshape((sents_mask.shape[0]*sents_mask.shape[1],sents_mask.shape[2])),
             mask_matrix_r = T.repeat(claim_mask,cand_size, axis=0),
             image_shape=(batch_size*cand_size, 1, emb_size, sent_len),
             image_shape_r = (batch_size*cand_size, 1, emb_size, claim_len),
             filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]),
             filter_shape_context=(hidden_size[0], 1,emb_size, 1),
             W=att_conv_W, b=att_conv_b,
             W_context=conv_W_context, b_context=conv_b_context)
    attentive_sent_embeddings_l = attentive_conv_layer.attentive_maxpool_vec_l.reshape((batch_size, cand_size, hidden_size[0]))  #(batch_size*cand_size, hidden_size)
    attentive_sent_embeddings_r = attentive_conv_layer.attentive_maxpool_vec_r.reshape((batch_size, cand_size, hidden_size[0]))
    masked_sents_attconv = attentive_sent_embeddings_l*sents_labels.dimshuffle(0,1,'x')
    masked_claim_attconv = attentive_sent_embeddings_r*sents_labels.dimshuffle(0,1,'x')
    fine_max = T.concatenate([T.max(masked_sents_attconv, axis=1),T.max(masked_claim_attconv, axis=1)],axis=1) #(batch, 2*hidden)
    # fine_sum = T.concatenate([T.sum(masked_sents_attconv, axis=1),T.sum(masked_claim_attconv, axis=1)],axis=1) #(batch, 2*hidden)
    "Logistic Regression layer"
    joint_LR_input = T.concatenate([premise_hypo_emb,fine_max], axis=1)
    joint_LR_input_size=2*hidden_size[0]+2*hidden_size[0]

    joint_U_a = create_ensemble_para(rng, 3, joint_LR_input_size) # (input_size, 3)
    joint_LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    joint_LR_para=[joint_U_a, joint_LR_b]

    joint_layer_LR=LogisticRegression(rng, input=joint_LR_input, n_in=joint_LR_input_size, n_out=3, W=joint_U_a, b=joint_LR_b) #basically it is a multiplication between weight matrix and input feature vector
    # joint_loss=joint_layer_LR.negative_log_likelihood(joint_labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.


    params = [init_embeddings]+NN_para+LR_para + joint_LR_para
    print('initialze model parameters...')
    load_model_from_file('/home1/w/wenpeng/dataset/FEVER/model_para_0.9936287838053803', params)

    # train_model = theano.function([sents_ids,sents_mask,sents_labels,claim_ids,claim_mask,joint_sents_ids,joint_sents_mask,joint_sents_labels, joint_claim_ids, joint_claim_mask, joint_labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([sents_ids,sents_mask, claim_ids,claim_mask], [inter_matrix,binarize_prob,joint_layer_LR.y_pred], allow_input_downcast=True, on_unused_input='ignore')
    # dev_model = theano.function([sents_ids,sents_mask, claim_ids,claim_mask], [binarize_prob,joint_layer_LR.y_pred], allow_input_downcast=True, on_unused_input='ignore')

    ###############
    # TRAIN MODEL #
    ###############
    print('... testing')
    # early-stopping parameters

    batch_score_vec, batch_binary_vec, pred_i=test_model(
            test_sents,
            test_sent_masks,
            test_claims,
            test_claim_mask
            )
    sorted_indices = np.argsort(batch_score_vec[0])[::-1] #descending order
    selected_sents = []
    for index in sorted_indices:
        if batch_binary_vec[0][index] == 1:
            selected_sents.append(sent_cand_list[index])
            if len(selected_sents)==5:
                break


    # for i, indicator in enumerate(list(batch_binary_vec[0])):
    #     if indicator == 1:
    #         selected_sents.append(sent_cand_list[i])
    return pred_id2label.get(pred_i[0]) +'"<p>"'+'"<br />"'.join(selected_sents)+'"<p/>"'
    # print( pred_id2label.get(pred_i[0]) +'<br />'+ '<br />'.join(selected_sents))




if __name__ == '__main__':
    root = '/home1/w/wenpeng/dataset/FEVER/'
    with open(root+'title2sentlist.p', 'rb') as fp:
        print('open title2sentlist.p...')
        title2sentlist = pickle.load(fp)
    with open(root+'title2wordlist.p', 'rb') as fp:
        print('open title2wordlist.p...')
        title2wordlist = pickle.load(fp)
    print(evaluate_lenet5('Dan Roth works in University of Illinois at Urbana-Champaign',title2sentlist,title2wordlist))
