import os
import numpy as np
import argparse
import Parameters
from Bio import SeqIO
#from fastText import train_unsupervised,load_model
from nltk import ngrams

class prepareData():
    def __init__(self):
        #self.encoding = "one-hot"
        self.encoding = "pssm"
        #self.encoding = "word2vec"
        self.PSSM_HOME = '../seq_mat/'
        self.sequence_length = Parameters.sequence_length
        self.blosum62 = {}
        blosum_reader = open('BLOSUM62.txt', 'r')
        self.charmap = {}
        self.index2char = {}
        #self.f = load_model("bio_protein_embedding.bin")
        count = 0
        # read the matrix of blosum62
        for line in blosum_reader:
            count = count + 1
            if count <= 7:
                continue
            line = line.strip('\r').split()
            array = [float(x) for x in line[1:21]]
            #array.sort(reverse=True)
            #array[4:] = [0.0] * 16
            self.blosum62[line[0]] = array#[float(x) for x in line[1:21]]

    def readPSSM(self, name):
        pssm_path = self.PSSM_HOME + name + '.fa_mat'
        #print(pssm_path)
        pssm_reader = open(pssm_path, 'r')
        #print(pssm_path)

        count = 0
        mat = []
        # read pssm file
        index = 0
        #if(len(pssm_reader)<=1):
        #    return mat

        for line in pssm_reader:
            count = count + 1
            if count <= 3:
                continue
            # print line
            #waring I did not get this line
            #if cmp(line, '\n') == 0:
            #    break
            #which indicates the end of PSSM
            if (line)=='\n':
                break
            line = line.strip('\n').split()
            # print line
            row = [float(k) for k in line[2:22]]
            #row.sort(reverse=True)
            #row[4:] = [0.0] * 16
            row2 = [0 if abs(k) < 3 else k for k in row]
            mat.append(row2)
            #print(row)
        #print("end")        

        #convert chap map to one-hot matrix

        '''
        >>> a = np.array([1, 0, 3])
        >>> b = np.zeros((3, 4))
        >>> b[np.arange(3), a] = 1
        >>> b
        array([[ 0.,  1.,  0.,  0.],
               [ 1.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  1.]])
        
        #warning: why S and X get same 0 index
        #To do: fix the one-hot encoding
        one_hot_mat = []
        for line in pssm_reader:
            count = count + 1
            if count <= 3:
                continue
            # print line
            #waring I did not get this line
            #if cmp(line, '\n') == 0:
            #    break
            #which indicates the end of PSSM
            if (line)=='\n':
                break
            line = line.strip('\n').split()
            # print line

            index = self.charmap[line[1]]

            row = [0 for k in range(len(self.charmap))]
            row[index] = 1
            
            # print 'len_row: ', len(row)
            one_hot_mat.append(row)
            #print(row)
        '''
        #print("there",mat[0][0][0])
        
        pssm_reader.close()
        return mat

    #build the char map for all the training data
    def read_one_hot(self, path1,path2,path3,path4):
        index = 0
        record = list(SeqIO.parse(path1, 'fasta'))

        for seq in record:
            #each character
            for aa in seq.seq:
                if aa not in self.charmap:
                    self.charmap[aa.upper()] = index
                    index = index +1    

        record = list(SeqIO.parse(path2, 'fasta'))
        for seq in record:
            #each character
            for aa in seq.seq:
                if aa not in self.charmap:
                    self.charmap[aa.upper()] = index
                    index = index +1    
        record = list(SeqIO.parse(path3, 'fasta'))
        for seq in record:
            #each character
            for aa in seq.seq:
                if aa not in self.charmap:
                    self.charmap[aa.upper()] = index
                    index = index +1    
        record = list(SeqIO.parse(path4, 'fasta'))
        for seq in record:
            #each character
            for aa in seq.seq:
                if aa not in self.charmap:
                    self.charmap[aa.upper()] = index
                    index = index +1    
        
        self.index2char = {v: k for k, v in self.charmap.items()}

        return self.charmap

    def getProteinBlosum(self, protein):
        protein_lst = []
        for aa in protein.seq:
            aa = aa.upper()
            protein_lst.append(self.blosum62[aa])
        return protein_lst

    #this could be replaced by word2vec
    #also need to be compared with the one-hot encoding
    def genertateMat(self, path):
        protein_names = []
        record = list(SeqIO.parse(path, 'fasta'))

        mats = np.asarray(
            np.zeros([len(record), self.sequence_length, 20]))

        one_hot_mats = np.asarray(
            np.zeros([len(record), self.sequence_length, len(self.charmap)])) #one-hot encoding representation

        word2vec_mats = np.asarray(
            np.zeros([len(record), self.sequence_length, 20])) #one-hot encoding representation

        index = 0
        count = 0
        for seq in record:
            #print('seq.id:',seq.id)
            #mat = self.readPSSM(seq.id)
            
            protein_names.append(seq.id)
            #generate the one-hot encoding for this seq
            one_hot_mat = []
            #generate word2vec encoding 
            word2vec_mat = []
           
            #extend to ngram version
            for aa in seq.seq:
                map_index = self.charmap[aa.upper()]
                row = [0 for k in range(len(self.charmap))]
                row[map_index] = 1
                    
                # print 'len_row: ', len(row)
                one_hot_mat.append(row)
                word2vec_mat.append(self.blosum62[aa])# i do not think we will have oov

            if not os.path.exists(self.PSSM_HOME + seq.id +'.fa_mat'):
                count = count + 1
                #print ('not exist for seq id', seq.id)
                # for proteins cannot generate PSSM, using blosum62 instead
                mat = self.getProteinBlosum(seq)
            else:
                mat = self.readPSSM(seq.id)

            mat = np.asarray(mat)
            #print("mat shape", mat.shape)
            one_hot_mat =  np.asarray(one_hot_mat)
            word2vec_mat = np.asarray(word2vec_mat)

            #print(mat.shape,one_hot_mat.shape)

            #assert len(mat)==len(one_hot_mat) #why not equal?
            #print('len of mat',len(mat))
            #print('len of one hot mat', len(one_hot_mat))

            if len(mat)!=len(one_hot_mat):
                continue
            else:
                seqLength = mat.shape[0]
                #print (mat.shape)
                if seqLength > self.sequence_length:
                    seqLength = self.sequence_length

                inputVector= np.zeros([self.sequence_length, 20])
                #doing the padding staff
                for i in range(seqLength):
                    inputVector[i] = mat[i]

                inputVector_one_hot= np.zeros([self.sequence_length, len(self.charmap)])
                #doing the padding staff
                for i in range(seqLength):
                    inputVector_one_hot[i] = one_hot_mat[i]

                inputVector_word2vec= np.zeros([self.sequence_length, 20])#for word2vec
                #doing the padding staff
                for i in range(seqLength):
                    inputVector_word2vec[i] = word2vec_mat[i]

                mats[index] = inputVector

                one_hot_mats[index] = inputVector_one_hot

                word2vec_mats[index] = inputVector_word2vec
                index = index + 1
        #print("mats",mats)
        #print("word_vec_mats",word2vec_mats)

        if self.encoding == "one-hot":
            return (one_hot_mats,protein_names)
        elif self.encoding =="word2vec":
            return (word2vec_mats,protein_names)
        else:
            return (mats,protein_names)


    #kmer-word-embedding:
    def genertate_kmer(self, path):
        protein_names = []
        record = list(SeqIO.parse(path, 'fasta'))

        word2vec_mats = np.asarray(
            np.zeros([len(record), self.sequence_length, 100])) #one-hot encoding representation

        index = 0
        count = 0

        f = load_model("bio_protein_embedding_3gram.bin")
        for seq in record:
            #print('seq.id :', seq.id)

            protein_names.append(seq.id)
            #generate word2vec encoding 
            word2vec_mat = []
           
            #extend to ngram version

            #split it into ngrams
            temp_seq = []
            for aa in seq.seq:
                # print(aa)
                temp_seq.append(aa)  # we can also extend to kmer-version
            # for kmer version we need to first split into kmer
            temp_seq_str = " ".join(temp_seq)

            ngrams_temp = ngrams(temp_seq_str.split(), 3)
            ngrams_string = [ ''.join(ele) for ele in ngrams_temp] #do we need to do this?
            #print(seq.seq)
            #print(ngrams_string)
            #for aa in ngrams_string:
                #up
                #print(aa)
                #word2vec_mat.append(f.get_word_vector(aa))# i do not think we will have oov
                                                               # extract from the ngram model

            word2vec_mat = np.asarray(word2vec_mat)

            #print(mat.shape,one_hot_mat.shape)

            #assert len(mat)==len(one_hot_mat) #why not equal?

            if len(word2vec_mat)==0:
                continue
            else:
                seqLength = word2vec_mat.shape[0]
                #print (word2vec_mat.shape)
                if seqLength > self.sequence_length:
                    seqLength = self.sequence_length
                inputVector= np.zeros([self.sequence_length, 100])
                #doing the padding staff
                for i in range(seqLength):
                    inputVector[i] = word2vec_mat[i]
                word2vec_mats[index] = inputVector
                index = index + 1

            return (word2vec_mats,protein_names)

    def generateInputData(self, args):
        self.read_one_hot(args.pos_train_dir,args.neg_train_dir,args.pos_test_dir,args.neg_test_dir)
        pos_train_mat,pos_train_names = self.genertateMat(args.pos_train_dir)
        print("here",pos_train_mat)

        neg_train_mat,neg_train_names = self.genertateMat(args.neg_train_dir)
        pos_test_mat,pos_test_names = self.genertateMat(args.pos_test_dir)
        neg_test_mat,neg_test_names = self.genertateMat(args.neg_test_dir)
        pos_train_num = pos_train_mat.shape[0]
        neg_train_num = neg_train_mat.shape[0]
        pos_test_num = pos_test_mat.shape[0]
        neg_test_num = neg_test_mat.shape[0]

        print(pos_train_mat.shape, neg_train_mat.shape)

        train_mat = np.vstack((pos_train_mat, neg_train_mat))
        test_mat = np.vstack((neg_test_mat,pos_test_mat))
        train_label = np.hstack((np.ones(pos_train_num), np.zeros(neg_train_num)))
        test_label = np.hstack((np.zeros(neg_test_num),np.ones(pos_test_num)))
        pos_test_names.extend(neg_test_names)
        test_names = pos_test_names

        print(self.charmap.keys())
        print(len(self.charmap))
        print(self.blosum62.keys())
        print(len(self.blosum62))

        diff_value = set(self.blosum62.keys())-set(self.charmap.keys())
        print (diff_value)
        print(self.charmap)        
        print (self.index2char)

        return (train_mat, train_label, test_mat, test_label, test_names)

    #to generate k_mer embedding
    def generateInputData_kmer(self,args):
        print (args.pos_train_dir)
        pos_train_mat,pos_train_names = self.genertate_kmer(args.pos_train_dir)
        neg_train_mat,neg_train_names = self.genertate_kmer(args.neg_train_dir)
        pos_test_mat,pos_test_names = self.genertate_kmer(args.pos_test_dir)
        neg_test_mat,neg_test_names = self.genertate_kmer(args.neg_test_dir)
        pos_train_num = pos_train_mat.shape[0]
        neg_train_num = neg_train_mat.shape[0]
        pos_test_num = pos_test_mat.shape[0]
        neg_test_num = neg_test_mat.shape[0]

        print(pos_train_mat.shape, neg_train_mat.shape)

        train_mat = np.vstack((pos_train_mat, neg_train_mat))
        test_mat = np.vstack((neg_test_mat,pos_test_mat))
        train_label = np.hstack((np.ones(pos_train_num), np.zeros(neg_train_num)))
        test_label = np.hstack((np.zeros(neg_test_num),np.ones(pos_test_num)))
        pos_test_names.extend(neg_test_names)
        test_names = pos_test_names

        return (train_mat, train_label, test_mat, test_label, test_names)
        
    def generateTestingSamples(self, args):

        pos_test_mat, pos_test_names = self.genertateMat(args.pos_test_dir)
        neg_test_mat, neg_test_names = self.genertateMat(args.neg_test_dir)

        pos_test_num = pos_test_mat.shape[0]
        neg_test_num = neg_test_mat.shape[0]

        test_mat = np.vstack((pos_test_mat, neg_test_mat))

        test_label = np.hstack((np.ones(pos_test_num), np.zeros(neg_test_num)))
        pos_test_names.extend(neg_test_names)
        test_names = pos_test_names
        return (test_mat, test_label, test_names)