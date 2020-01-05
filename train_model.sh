python train.py -e 0.0005 -family_index c.37.1.19 -train True -pos_train_dir data/pos-train.c.37.1.19.fasta -neg_train_dir data/neg-train.c.37.1.19.fasta -pos_test_dir data/pos-test.c.37.1.19.fasta -neg_test_dir data/neg-test.c.37.1.19.fasta

python train.py -e 0.0005 -family_index a.138.1.3 -train True -pos_train_dir data/pos-train.a.138.1.3.fasta -neg_train_dir data/neg-train.a.138.1.3.fasta -pos_test_dir data/pos-test.a.138.1.3.fasta -neg_test_dir data/neg-test.a.138.1.3.fasta

#for extended scop dataset
python train.py -e 0.0005 -family_index 2.1.1.1 -train True -pos_train_dir datasetsSCOP1.53/positive-training-extended/pos-train.2.1.1.1.fasta  -neg_train_dir datasetsSCOP1.53/original-dataset/neg-train.2.1.1.1.fasta -pos_test_dir datasetsSCOP1.53/original-dataset/pos-test.2.1.1.1.fasta -neg_test_dir datasetsSCOP1.53/original-dataset/neg-test.2.1.1.1.fasta

python train.py -e 0.0005 -family_index 2.1.1.1 -train True -pos_train_dir datasetsSCOP1.53/original-dataset/pos-train.2.1.1.1.fasta  -neg_train_dir datasetsSCOP1.53/original-dataset/neg-train.2.1.1.1.fasta -pos_test_dir datasetsSCOP1.53/original-dataset/pos-test.2.1.1.1.fasta -neg_test_dir datasetsSCOP1.53/original-dataset/neg-test.2.1.1.1.fasta
