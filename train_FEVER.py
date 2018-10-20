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

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from theano.tensor.signal import downsample
from random import shuffle
from theano.tensor.nnet.bn import batch_normalization

from load_data import load_fever_train,  load_fever_dev
from common_functions import store_model_to_file,Attentive_Conv_for_Pair_easy_version,load_word2vec,load_word2vec_to_init, store_model_to_file, elementwise_is_two,Conv_with_Mask_with_Gate, Conv_with_Mask, create_conv_para, L2norm_paraList, create_HiddenLayer_para, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para

from preprocess import compute_f1_two_list_names

'''
0.355620320616(f1) 0.513917907171(recall) 0.325235311538  ;
'''

def evaluate_lenet5(learning_rate=0.02, n_epochs=100, emb_size=300, batch_size=50, filter_size=[3], sent_len=40, claim_len=20, cand_size=10,hidden_size=[300,300], max_pred_pick=5):

    model_options = locals().copy()
    print "model options", model_options

    seed=1234
    np.random.seed(seed)
    rng = np.random.RandomState(seed)    #random seed, control the model generates the same results
    srng = T.shared_randomstreams.RandomStreams(rng.randint(seed))

    "load raw data"
    train_sents, train_sent_masks, train_sent_labels, train_claims, train_claim_mask, _, word2id  = load_fever_train(sent_len, claim_len, cand_size)
    test_sents, test_sent_masks, test_sent_labels, test_claims, test_claim_mask, test_sent_names,test_ground_names,_, word2id = load_fever_dev(sent_len, claim_len, cand_size, word2id)

    train_sents=np.asarray(train_sents, dtype='int32')
    # dev_sents_l=np.asarray(all_sentences_l[1], dtype='int32')
    test_sents=np.asarray(test_sents, dtype='int32')

    train_sent_masks=np.asarray(train_sent_masks, dtype=theano.config.floatX)
    # dev_masks_l=np.asarray(all_masks_l[1], dtype=theano.config.floatX)
    test_sent_masks=np.asarray(test_sent_masks, dtype=theano.config.floatX)

    train_sent_labels=np.asarray(train_sent_labels, dtype='int32')
    # dev_sents_r=np.asarray(all_sentences_r[1]    , dtype='int32')
    # test_sent_labels=np.asarray(test_sent_labels, dtype='int32')

    train_claims=np.asarray(train_claims, dtype='int32')
    # dev_sents_r=np.asarray(all_sentences_r[1]    , dtype='int32')
    test_claims=np.asarray(test_claims, dtype='int32')

    train_claim_mask=np.asarray(train_claim_mask, dtype=theano.config.floatX)
    # dev_masks_r=np.asarray(all_masks_r[1], dtype=theano.config.floatX)
    test_claim_mask=np.asarray(test_claim_mask, dtype=theano.config.floatX)


    # train_labels_store=np.asarray(all_labels[0], dtype='int32')
    # dev_labels_store=np.asarray(all_labels[1], dtype='int32')
    # test_labels_store=np.asarray(all_labels[2], dtype='int32')

    train_size=len(train_claims)
    # dev_size=len(dev_labels_store)
    test_size=len(test_claims)
    print 'train size: ', train_size, ' test size: ', test_size

    vocab_size=len(word2id)+1

    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_word2vec()
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    init_embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    "now, start to build the input form of the model"
    sents_ids=T.itensor3() #(batch, cand_size, sent_len)
    sents_mask=T.ftensor3()
    sents_labels=T.imatrix() #(batch, cand_size)
    claim_ids = T.imatrix() #(batch, claim_len)
    claim_mask = T.imatrix()
    # labels=T.ivector()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'


    embed_input_sents=init_embeddings[sents_ids.flatten()].reshape((batch_size*cand_size, sent_len, emb_size)).dimshuffle(0,2,1)#embed_input(init_embeddings, sents_ids_l)#embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    embed_input_claim=init_embeddings[claim_ids.flatten()].reshape((batch_size,claim_len, emb_size)).dimshuffle(0,2,1)



    conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    # conv_W2, conv_b2=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[1]))
    NN_para = [conv_W, conv_b]
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

    concate_claim_sent = T.concatenate([batch_claim_emb,batch_sent_emb, T.sum(batch_claim_emb*batch_sent_emb, axis=2).dimshuffle(0,1,'x') ], axis=2)
    concate_2_matrix = concate_claim_sent.reshape((batch_size*cand_size, hidden_size[0]*2+1))

    LR_input = concate_2_matrix#T.concatenate([sent_embeddings,sent_embeddings2], axis=1)
    LR_input_size = hidden_size[0]*2+1
    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    U_a = create_ensemble_para(rng, 1, LR_input_size) # the weight matrix hidden_size*2
    # LR_b = theano.shared(value=np.zeros((8,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR_para=[U_a]
    # layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=8, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    score_matrix = T.nnet.sigmoid(concate_2_matrix.dot(U_a))  #batch * 12
    inter_matrix = score_matrix.reshape((batch_size, cand_size))

    # inter_sent_claim = T.batched_dot(batch_sent_emb, batch_claim_emb) #(batch_size, cand_size, 1)
    # inter_matrix = T.nnet.sigmoid(inter_sent_claim.reshape((batch_size, cand_size)))
    '''
    maybe 1.0-inter_matrix can be rewritten into 1/e^(inter_matrix)
    '''
    # prob_pos = T.where( sents_labels < 1, 1.0-inter_matrix, inter_matrix)
    # loss = -T.mean(T.log(prob_pos))
    #f1 as loss
    batch_overlap = T.sum(sents_labels*inter_matrix, axis=1)
    batch_recall = batch_overlap / T.sum(sents_labels, axis=1)
    batch_precision = batch_overlap / T.sum(inter_matrix, axis=1)
    batch_f1 = 2.0*batch_recall*batch_precision/(batch_recall+batch_precision)
    loss = -T.mean(T.log(batch_f1))#+1e-3*((U_a**2).sum()+(conv_W**2).sum())
    #
    # "Logistic Regression layer"
    # LR_input = T.concatenate([attentive_sent_embeddings_l,attentive_sent_embeddings_r,attentive_sent_embeddings_l+attentive_sent_embeddings_r,attentive_sent_embeddings_l*attentive_sent_embeddings_r],axis=1)
    # LR_input_size=4*hidden_size[0]
    #
    # U_a = create_ensemble_para(rng, 3, LR_input_size) # (input_size, 3)
    # LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    # LR_para=[U_a, LR_b]
    #
    # layer_LR=LogisticRegression(rng, input=normalize_matrix_col_wise(LR_input), n_in=LR_input_size, n_out=3, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    # loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.

    '''
    testing
    '''
    binarize_prob = T.where( inter_matrix > 0.5, 1, 0)  #(batch_size, cand_size



    params = [init_embeddings]+NN_para+LR_para
    cost=loss
    "Use AdaGrad to update parameters"
    updates =   Gradient_Cost_Para(cost,params, learning_rate)


    train_model = theano.function([sents_ids,sents_mask,sents_labels,claim_ids,claim_mask], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    # dev_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([sents_ids,sents_mask,claim_ids,claim_mask], inter_matrix, allow_input_downcast=True, on_unused_input='ignore')

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50000000000  # look as this many examples regardless
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False

    n_train_batches=train_size/batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]
    # n_dev_batches=dev_size/batch_size
    # dev_batch_start=list(np.arange(n_dev_batches)*batch_size)+[dev_size-batch_size]
    n_test_batches=test_size/batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]


    max_acc_dev=0.0
    max_test_f1=0.0

    cost_i=0.0
    train_indices = range(train_size)

    while epoch < n_epochs:
        epoch = epoch + 1

        random.Random(100).shuffle(train_indices) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed

        iter_accu=0

        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_batch = train_indices[batch_id:batch_id+batch_size]
            '''
            train_sents, train_sent_masks, train_sent_labels, train_claims, train_claim_mask
            sents_ids,sents_mask,sents_labels,claim_ids,claim_mask
            '''
            cost_i+= train_model(
                                train_sents[train_id_batch],
                                train_sent_masks[train_id_batch],
                                train_sent_labels[train_id_batch],
                                train_claims[train_id_batch],
                                train_claim_mask[train_id_batch])

            #after each 1000 batches, we test the performance of the model on all test data
            # if (epoch==1 and iter%1000==0) or (epoch>=2 and iter%5==0):
            if iter%10==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()



                '''
                test
                test_sents, test_sent_masks, test_sent_labels, test_claims, test_claim_mask,
                sents_ids,sents_mask,claim_ids,claim_mask
                '''
                f1_sum=0.0
                recall_sum = 0.0
                precision_sum = 0.0
                for test_batch_id in test_batch_start: # for each test batch
                    batch_prob=test_model(
                            test_sents[test_batch_id:test_batch_id+batch_size],
                            test_sent_masks[test_batch_id:test_batch_id+batch_size],
                            test_claims[test_batch_id:test_batch_id+batch_size],
                            test_claim_mask[test_batch_id:test_batch_id+batch_size])

                    batch_sent_labels = test_sent_labels[test_batch_id:test_batch_id+batch_size]
                    batch_sent_names = test_sent_names[test_batch_id:test_batch_id+batch_size]
                    batch_ground_names = test_ground_names[test_batch_id:test_batch_id+batch_size]


                    for i in range(batch_size):
                        pred_sent_names = []
                        gold_sent_names = batch_ground_names[i]

                        zipped=[(batch_prob[i,k],batch_sent_labels[i][k],batch_sent_names[i][k]) for k in range(cand_size)]
                        sorted_zip = sorted(zipped, key=lambda x: x[0], reverse=True)
                        # print 'sorted_zip:', sorted_zip
                        # exit(0)
                        for j in range(cand_size):
                            triple = sorted_zip[j]
                            if triple[1] == 1.0:
                                '''
                                we should consider a rank, instead of binary
                                '''
                                if triple[0] >0.5:
                                    pred_sent_names.append(triple[2])
                                    # if len(pred_sent_names) == max_pred_pick:
                                    #     break
                        f1_i, recall_i, precision_i = compute_f1_two_list_names(pred_sent_names, gold_sent_names)
                        # print f1_i, recall_i, precision_i
                        f1_sum+=f1_i
                        recall_sum+=recall_i
                        precision_sum+=precision_i


                test_f1=f1_sum/(len(test_batch_start)*batch_size)
                test_recall = recall_sum/(len(test_batch_start)*batch_size)
                test_precision = precision_sum/(len(test_batch_start)*batch_size)

                if test_f1 > max_test_f1:
                    max_test_f1=test_f1
                print '\t\tcurrent test_f1:', test_f1,test_recall,test_precision, ' ; ','\t\t\t\t\tmax_test_f1:', max_test_f1



        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return max_acc_test



if __name__ == '__main__':
    evaluate_lenet5()
