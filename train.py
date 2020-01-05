import numpy as np
import torch
from util import *
import os
import argparse
from Data import *
from models import NLINet
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, auc
import re
import os
import sys
import time

def trainepoch(epoch,nli_net,s1,label,data_helper,optimizer,loss_fn):
    #have not shuffled yet
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(s1))#have to shuffle, otherwise all 0 or 1
    s1 = s1[permutation]
    label = label[permutation]

    print (s1.shape)
    print(label.shape)

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        #s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
        #                             word_vec)
        s1_batch, s1_len = get_batch_pssm(s1[stidx:stidx + params.batch_size])
       
        # To do adjust the require_gradient
        # we may not update here? Should be in the embedding matrix?
        #default if false
        s1_batch = Variable(s1_batch.cuda(),requires_grad=True)
        #s1_batch = Variable(s1_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(label[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.size(1)  # actual batch size

        #print("input requires_grad: ",s1_batch.requires_grad)

        nli_net.encoder.convnet1[0].weight.data[0:40,:,5] = torch.tensor(0.0)
        nli_net.encoder.convnet1[0].weight.data[40:50,:,4] = torch.tensor(0.0)
        nli_net.encoder.convnet1[0].weight.data[50:60,:,3] = torch.tensor(0.0)
        nli_net.encoder.convnet1[0].weight.data[60:70,:,2] = torch.tensor(0.0)
        nli_net.encoder.convnet1[0].weight.data[70:80,:,1] = torch.tensor(0.0)

        # model forward
        output = nli_net((s1_batch, s1_len))

        #print("Output shape:", output)

        pred = output.data.max(1)[1]
        #print(pred)
        #print(F.softmax(output))
        #print(pred.shape,(F.softmax(output,dim=1).data)[:,1].shape)

        #print(pred,tgt_batch)
        #For Roc we might need the softmax output
        #this is the positive class prediction score
        if stidx!=0:
            prev_pred = torch.cat((prev_pred, (F.softmax(output,dim=1).data)[:,1]), 0)
        else:
            prev_pred = (F.softmax(output,dim=1).data)[:,1]

        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data)
        words_count += (s1_batch.nelement() ) / params.word_emb_dim

        #print (output.data[0]) #warning #output summ not one?

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                #try:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
                #except:
                #None
                #p._grad.data.div_(k)
                #total_norm += p._grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm.cpu().numpy())

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm

        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if stidx == 0 and epoch%20 == 0:
            print("after")
            print(nli_net.encoder.convnet1[0].weight.data)

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs,dtype=np.float32), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            np.round(100.*correct/(stidx+k), 2))) #tensor do not have round #what is the type of correct
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = np.round(100 * correct/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))

    return train_acc,prev_pred,round(np.mean(all_costs,dtype=np.float32), 3)

def evaluate(epoch, nli_net1, nli_net2, nli_net3, nli_net4 ,s1,label, data_helper,writer,eval_type='valid', final_eval=False):
    nli_net1.eval()
    nli_net2.eval()
    nli_net3.eval()
    nli_net4.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = s1 if eval_type == 'valid' else test['s1']
   
    target = label if eval_type == 'valid' else test['label']

    print (s1.shape)
    print (target.shape)

    # using writer to track the embedding
    #I want to check the discriminative information
    #only vis it in the last epoch, when it is almost done
    if epoch == params.n_epochs:
        vis_batch, vis_len = get_batch_pssm(s1)
        vis_batch = Variable(vis_batch.cuda(),requires_grad=True)
        tgt_batch = Variable(torch.LongTensor(label)).cuda()

        #which one works better?
        #second one should get the before softmax?
        #sequence_embedding = nli_net.encode((vis_batch,vis_len))
        sequence_embedding1 = nli_net1((vis_batch,vis_len))
        sequence_embedding2 = nli_net2((vis_batch,vis_len))
        sequence_embedding3 = nli_net3((vis_batch,vis_len))
        sequence_embedding4 = nli_net4((vis_batch,vis_len))

        print(sequence_embedding1.shape)
        print(tgt_batch.data)
        writer.add_embedding(
                    sequence_embedding1,
                    metadata=tgt_batch.data
        )
    
    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch_pssm(s1[i:i + params.batch_size])
        
        s1_batch = Variable(s1_batch.cuda(),requires_grad=True)
        tgt_batch = Variable(torch.LongTensor(label[i:i + params.batch_size])).cuda()

        # model forward
        output1 = nli_net1((s1_batch, s1_len)) #why do we need the length, becasue encoder need it
        pred1 = output1.data.max(1)[1]
        output2 = nli_net2((s1_batch, s1_len)) #why do we need the length, becasue encoder need it
        pred2 = output2.data.max(1)[1]
        output3 = nli_net3((s1_batch, s1_len)) #why do we need the length, becasue encoder need it
        pred3 = output3.data.max(1)[1]
        output4 = nli_net4((s1_batch, s1_len)) #why do we need the length, becasue encoder need it
        pred4 = output4.data.max(1)[1]

        #results = results + pred
        #print (pred.shape)
        if i!=0:
            prev_pred1 = torch.cat((prev_pred1, (F.softmax(output1,dim=1).data)[:,1]), 0)
        else:
            prev_pred1 = (F.softmax(output1,dim=1).data)[:,1]
        if i!=0:
            prev_pred2 = torch.cat((prev_pred2, (F.softmax(output2,dim=1).data)[:,1]), 0)
        else:
            prev_pred2 = (F.softmax(output2,dim=1).data)[:,1]
        if i!=0:
            prev_pred3 = torch.cat((prev_pred3, (F.softmax(output3,dim=1).data)[:,1]), 0)
        else:
            prev_pred3 = (F.softmax(output3,dim=1).data)[:,1]
        if i!=0:
            prev_pred4 = torch.cat((prev_pred4, (F.softmax(output4,dim=1).data)[:,1]), 0)
        else:
            prev_pred4 = (F.softmax(output4,dim=1).data)[:,1]
        
        #print('pred4',pred4)
        #print('pred3',pred3)
        #print('pred2',pred2)
        #print('pred1',pred1)
        #print('pred1 size',pred1.size())

        pred = torch.mean(torch.stack([pred1.float(),pred2.float(), pred3.float(),pred4.float()]),dim=0)
        pred[pred>=0.5] = 1
        pred[pred<0.5] = 0
        #print('pred size', pred.size())
        #print('pred',pred)
        
        #print('prev_pred4',prev_pred4)
        #print('prev pred3',prev_pred3)
        #print('prev pred2',prev_pred2)
        #print('prev_pred1',prev_pred1)
        #print('prev pred1 size',prev_pred1.size())

        prev_pred = torch.mean(torch.stack([prev_pred1,prev_pred2,prev_pred3,prev_pred4]),dim=0)
        #print('prev pred size', prev_pred.size())
        #print('prev pred',prev_pred)
        #prev_pred[prev_pred>=0.5] = 1
        #prev_pred[pred<0.5] = 0

                    #writer.add_embedding(
            #    out,
            #    metadata=label_batch.data,
            #    label_img=data_batch.data,
            #    global_step=n_iter)
        
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    eval_acc = np.round(100 * correct / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))
    '''
    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    '''
    return eval_acc,prev_pred

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    params = parseArguments(parser)
    params.n_epochs = 50
    params.gpu_id = 0
    print (params)
    threshold = params.e
    target_fam = params.family_index
    print("family_index: ",target_fam)

    data_helper = prepareData()
    if params.kmer!=True:
        train_mat, train_label, test_mat, test_label, test_names = data_helper.generateInputData(params)
    else:
        print('This is kmer embedding')
        train_mat, train_label, test_mat, test_label, test_names = data_helper.generateInputData_kmer(params)
    print(train_mat[0])   

    # set gpu device
    torch.cuda.set_device(params.gpu_id)

    """
    SEED
    """
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    #for the PSSM score #20
    params.word_emb_dim = train_mat.shape[2] #for one-hot encoding, to match the encoding dimension

    # model config
    config_nli_model = {
        'n_words'        :  0          , #seems not useful?#len(word_vec)
        'word_emb_dim'   :  params.word_emb_dim   ,
        'enc_lstm_dim'   :  params.enc_lstm_dim   ,
        'n_enc_layers'   :  params.n_enc_layers   ,
        'dpout_model'    :  params.dpout_model    ,
        'dpout_fc'       :  params.dpout_fc       ,
        'fc_dim'         :  params.fc_dim         ,
        'bsize'          :  params.batch_size     ,
        'n_classes'      :  params.n_classes      ,
        'pool_type'      :  params.pool_type      ,
        'nonlinear_fc'   :  params.nonlinear_fc   ,
        'encoder_type'   :  params.encoder_type   ,
        'use_cuda'       :  True                  ,
    }

    # model
    #encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
    #                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
    #                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
    #assert params.encoder_type in encoder_types, "encoder_type must be in " + \
    #                                             str(encoder_types)
    
    config_nli_model1 = config_nli_model
    config_nli_model2 = config_nli_model
    config_nli_model3 = config_nli_model
    config_nli_model4 = config_nli_model

    config_nli_model1['encoder_type'] = 'ConvNetEncoder1'
    config_nli_model2['encoder_type'] = 'ConvNetEncoder2'
    config_nli_model3['encoder_type'] = 'ConvNetEncoder3'
    config_nli_model4['encoder_type'] = 'ConvNetEncoder4'

    nli_net1 = NLINet(config_nli_model1)
    nli_net2 = NLINet(config_nli_model2)
    nli_net3 = NLINet(config_nli_model3)
    nli_net4 = NLINet(config_nli_model4)
    #print(nli_net)

    # loss
    weight = torch.FloatTensor(params.n_classes).fill_(1)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    loss_fn.size_average = False
    weight2 = torch.FloatTensor(params.n_classes).fill_(1)
    loss_fn2 = nn.CrossEntropyLoss(weight=weight2)
    loss_fn2.size_average = False
    weight3 = torch.FloatTensor(params.n_classes).fill_(1)
    loss_fn3 = nn.CrossEntropyLoss(weight=weight3)
    loss_fn3.size_average = False
    weight4 = torch.FloatTensor(params.n_classes).fill_(1)
    loss_fn4 = nn.CrossEntropyLoss(weight=weight4)
    loss_fn4.size_average = False

    #print('1')
    # optimizer
    optim_fn, optim_params = get_optimizer(params.optimizer)
    optimizer = optim_fn(nli_net1.parameters(), **optim_params)
    optim_fn2, optim_params2 = get_optimizer(params.optimizer)
    optimizer2 = optim_fn2(nli_net2.parameters(), **optim_params2)
    optim_fn3, optim_params3 = get_optimizer(params.optimizer)
    optimizer3 = optim_fn3(nli_net3.parameters(), **optim_params3)
    optim_fn4, optim_params4 = get_optimizer(params.optimizer)
    optimizer4 = optim_fn4(nli_net4.parameters(), **optim_params4)
    #print('2')
    # cuda by default
    nli_net1.cuda()
    nli_net2.cuda()
    nli_net3.cuda()
    nli_net4.cuda()
    loss_fn.cuda()
    loss_fn2.cuda()
    loss_fn3.cuda()
    loss_fn4.cuda()    

    """
    Train model on Remote Homology Detection Problem
    """
    epoch = 1
    best_roc = 0
    best_roc_50 = 0
    best_epoch = 1

    #keep track of the training loss

    train_losses = []
    roc_result=[]
    #print('3')

    writer = SummaryWriter(comment='mnist_embedding_training')
    #print('here')
    intial_loss = None

    while  epoch <= params.n_epochs:
        train_acc1,pre_results1,train_loss1 = trainepoch(epoch,nli_net1,train_mat, train_label,data_helper,optimizer,loss_fn)
        train_acc2,pre_results2,train_loss2 = trainepoch(epoch,nli_net2,train_mat, train_label,data_helper,optimizer2,loss_fn2)
        train_acc3,pre_results3,train_loss3 = trainepoch(epoch,nli_net3,train_mat, train_label,data_helper,optimizer3,loss_fn3)
        train_acc4,pre_results4,train_loss4 = trainepoch(epoch,nli_net4,train_mat, train_label,data_helper,optimizer4,loss_fn4)
    
        #print('train loss: ',train_loss)

        #if epoch == 1:
        #    intial_loss = train_loss
        #else:
        #    perc = (abs(intial_loss - train_loss)/intial_loss)
        #    print('percent ', perc)
        #    if perc>=0.50:
        #        break

        #roc = get_roc(train_label, pre_results, pre_results.shape[0])
        #roc50 = get_roc(train_label, pre_results, 50)
        #print (train_label.shape, pre_results.shape)
        #print (roc,roc50)
        
        #keep track of the training loss
        #train_losses.append(train_loss)
        #loss need to be torch variable
        #writer.add_scalar('train_loss', train_loss.data.item(), epoch)

        eval_acc,pre_results = evaluate(epoch, nli_net1, nli_net2, nli_net3, nli_net4,test_mat, test_label,data_helper,writer,'valid')
        #print(pre_results)
        #print(test_label)
        roc = get_roc(test_label, pre_results, pre_results.shape[0])
        roc50 = get_roc(test_label, pre_results, 50)
        #track the roc for test data
        roc_result.append(roc)
        print (test_label.shape, pre_results.shape)
        print (roc,roc50)

        if roc>best_roc:
            best_roc = roc
            best_roc_50 = roc50
            best_epoch = epoch
            #print('The best roc we have: %0.2f  %0.2f' %(best_roc,best_roc_50))

        epoch += 1

    #import matplotlib.pyplot as plt
    #print(train_losses)

    #plot the training loss and roc curve
    #print(roc_result)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(2,2,1)
    # ax1.plot(train_losses)
    # ax1.xlabel('Epochs')
    # ax1.ylabel('training loss')
    # ax1.title('Plot of training loss')

    # ax2 = fig.add_subplot(2,2,2)
    # ax2.plot(roc_result)
    # ax2.xlabel('Epochs')
    # ax2.ylabel('ROC')
    # ax2.title('Plot of ROC value for testing dataset')

    # plt.show()

    #plt.plot(train_losses)
   
    #plt.plot(train_losses)
    #plt.ylabel('training loss')
    #plt.xlabel('Epochs')
    #plt.title('Plot of training loss')
    #plt.show()

    
    #plt.plot(roc_result)
    #plt.ylabel('ROC')
    #plt.xlabel('Epochs')
    #plt.title('Plot of ROC value for testing dataset')
    #plt.show()

    def ROC_Plot(test_label,pre_results):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(test_label, pre_results)
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_label, pre_results)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        #import matplotlib.pyplot as plt
        #plt.figure()
        lw = 2
        #plt.plot(fpr[1], tpr[1], color='darkorange',
        #         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        #plt.xlim([0.0, 1.0])
        #plt.ylim([0.0, 1.05])
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.title('Receiver operating characteristic example')
        #plt.legend(loc="lower right")
        #plt.show()
    #ROC_Plot(test_label,pre_results)
    print('The best roc we have: %0.2f  %0.2f' %(best_roc,best_roc_50))

    with open(data_helper.encoding+params.encoder_type+str(params.n_epochs)+"WC_40-6_10-5_10-4_10-3_10-2"+"Results.txt", "a") as text_file:
         text_file.write(params.family_index+','+str(best_roc) +','+str(best_roc_50) + ',' +str(best_epoch) + "\n")