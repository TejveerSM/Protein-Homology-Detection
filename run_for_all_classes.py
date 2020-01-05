import os 

command = "python train.py -family_index 2.1.1.1 -train True -pos_train_dir datasetsSCOP1.53/positive-training-extended/pos-train.2.1.1.1.fasta  -neg_train_dir datasetsSCOP1.53/original-dataset/neg-train.2.1.1.1.fasta -pos_test_dir datasetsSCOP1.53/original-dataset/pos-test.2.1.1.1.fasta -neg_test_dir datasetsSCOP1.53/original-dataset/neg-test.2.1.1.1.fasta"
#os.system('python train.py -family_index 2.1.1.1')

import glob
file_names = glob.glob("SCOP167-superfamily/*.fasta")

new_file_names = []
family_indexes = []
for file in file_names:
	file = file.split("/")[1]
	#exclude one file
	if file!="pos--train.fasta":
		new_file_names.append(file)
		index = file.split('.')[1:5]
		family_indexes.append('.'.join(index))
print (new_file_names)
print (len(new_file_names))
print(set(family_indexes))
print(len(set(family_indexes)))


file_common_path = 'SCOP167-superfamily/'
family_indexes = set(family_indexes)
print("No of families", len(family_indexes))
for family in family_indexes:
	family_index  ='  -family_index '+family
	pos_train_dir ='  -pos_train_dir ' + file_common_path + 'pos-train.'+family+'.fasta'
	neg_train_dir ='  -neg_train_dir ' + file_common_path + 'neg-train.'+family+'.fasta'
	pos_test_dir  ='  -pos_test_dir ' + file_common_path + 'pos-test.'+family+'.fasta' 
	neg_test_dir = '  -neg_test_dir ' + file_common_path + 'neg-test.'+family+'.fasta' 
	print (family_index)
	print (pos_train_dir)
	print (neg_train_dir)
	os.system('python3 train.py'+family_index+pos_train_dir+neg_train_dir+pos_test_dir+neg_test_dir)