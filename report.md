embeding layers

    byt5 -1450 emb vector 6hr to genrate encoding for 
    |          | 47/188296 [00:05<6:04:28,  8.61it/s]

    canine -502 emb vector 1hr to genrate encoding for
    |          | 155/188296 [00:02<52:42, 59.49it/s]

    basic 128 dim embeding 5min and 96 percent accuracy




convolution layers -
50% dataset :- Epoch 6/6 | Train Loss: 0.1069, Train Acc: 0.9654 | Val Loss: 0.0921, Val Acc: 0.9704
100% dataset :-Epoch 6/6 | Train Loss: 0.0914, Train Acc: 0.9714 | Val Loss: 0.0771, Val Acc: 0.9749
    diffrent kernal in parallel:- 
        with 3 kernal + 5 kernal Epoch 6/6 | Train Loss: 0.0802, Train Acc: 0.9779 | Val Loss: 0.0635, Val Acc: 0.9805 in time 1.50 min

        with 3 kernal + 5 kernal + 7 kernal  

        ---------------
        
        ___________🧩 Using 50% of training data___________
        Epoch 1/6 | Train Loss: 0.2392, Train Acc: 0.9213 | Val Loss: 0.0975, Val Acc: 0.9699              
        Epoch 2/6 | Train Loss: 0.1184, Train Acc: 0.9667 | Val Loss: 0.0883, Val Acc: 0.9727              
        Epoch 3/6 | Train Loss: 0.0992, Train Acc: 0.9716 | Val Loss: 0.0735, Val Acc: 0.9767              
        Epoch 4/6 | Train Loss: 0.0895, Train Acc: 0.9735 | Val Loss: 0.0708, Val Acc: 0.9781              
        Epoch 5/6 | Train Loss: 0.0822, Train Acc: 0.9754 | Val Loss: 0.0652, Val Acc: 0.9799              
        Epoch 6/6 | Train Loss: 0.0790, Train Acc: 0.9762 | Val Loss: 0.0632, Val Acc: 0.9804              
        __________🧩 Using 100% of training data___________
        Epoch 1/6 | Train Loss: 0.1877, Train Acc: 0.9367 | Val Loss: 0.0938, Val Acc: 0.9707              
        Epoch 2/6 | Train Loss: 0.0986, Train Acc: 0.9711 | Val Loss: 0.0699, Val Acc: 0.9783              
        Epoch 3/6 | Train Loss: 0.0789, Train Acc: 0.9764 | Val Loss: 0.0617, Val Acc: 0.9800              
        Epoch 4/6 | Train Loss: 0.0723, Train Acc: 0.9782 | Val Loss: 0.0567, Val Acc: 0.9818              
        Epoch 5/6 | Train Loss: 0.0682, Train Acc: 0.9789 | Val Loss: 0.0584, Val Acc: 0.9817              
        Epoch 6/6 | Train Loss: 0.0643, Train Acc: 0.9796 | Val Loss: 0.0560, Val Acc: 0.9824              

        ✅ All datasets trained successfully!
        ---------------

        with 3 kernal + 5 kernal + 7 kernal +9 kernal  

            ___________🧩 Using 50% of training data___________
            Epoch 1/6 | Train Loss: 0.2469, Train Acc: 0.9078 | Val Loss: 0.1070, Val Acc: 0.9642              
            Epoch 2/6 | Train Loss: 0.1083, Train Acc: 0.9637 | Val Loss: 0.0763, Val Acc: 0.9749              
            Epoch 3/6 | Train Loss: 0.0875, Train Acc: 0.9725 | Val Loss: 0.0681, Val Acc: 0.9781              
            Epoch 4/6 | Train Loss: 0.0835, Train Acc: 0.9740 | Val Loss: 0.0660, Val Acc: 0.9781              
            Epoch 5/6 | Train Loss: 0.0750, Train Acc: 0.9762 | Val Loss: 0.0621, Val Acc: 0.9803              
            Epoch 6/6 | Train Loss: 0.0745, Train Acc: 0.9764 | Val Loss: 0.0633, Val Acc: 0.9804              
            __________🧩 Using 100% of training data___________
            Epoch 1/6 | Train Loss: 0.1611, Train Acc: 0.9487 | Val Loss: 0.0803, Val Acc: 0.9743              
            Epoch 2/6 | Train Loss: 0.0921, Train Acc: 0.9732 | Val Loss: 0.0704, Val Acc: 0.9773              
            Epoch 3/6 | Train Loss: 0.0813, Train Acc: 0.9757 | Val Loss: 0.0672, Val Acc: 0.9784              
            Epoch 4/6 | Train Loss: 0.0754, Train Acc: 0.9771 | Val Loss: 0.0612, Val Acc: 0.9814              
            Epoch 5/6 | Train Loss: 0.0702, Train Acc: 0.9783 | Val Loss: 0.0598, Val Acc: 0.9815              
            Epoch 6/6 | Train Loss: 0.0649, Train Acc: 0.9796 | Val Loss: 0.0567, Val Acc: 0.9825


    diffrent kernal size in sequence:-
     7kernal -> 5kernal-> 3kernal

        ___________🧩 Using 50% of training data___________
        Epoch 1/6 | Train Loss: 0.3103, Train Acc: 0.8794 | Val Loss: 0.1373, Val Acc: 0.9546              
        Epoch 2/6 | Train Loss: 0.1326, Train Acc: 0.9599 | Val Loss: 0.0987, Val Acc: 0.9681              
        Epoch 3/6 | Train Loss: 0.1082, Train Acc: 0.9685 | Val Loss: 0.0872, Val Acc: 0.9714              
        Epoch 4/6 | Train Loss: 0.0969, Train Acc: 0.9712 | Val Loss: 0.0774, Val Acc: 0.9745              
        Epoch 5/6 | Train Loss: 0.0923, Train Acc: 0.9721 | Val Loss: 0.0764, Val Acc: 0.9751              
        Epoch 6/6 | Train Loss: 0.0881, Train Acc: 0.9737 | Val Loss: 0.0787, Val Acc: 0.9749              
        __________🧩 Using 100% of training data___________
        Epoch 1/6 | Train Loss: 0.2202, Train Acc: 0.9172 | Val Loss: 0.1166, Val Acc: 0.9617              
        Epoch 2/6 | Train Loss: 0.1251, Train Acc: 0.9629 | Val Loss: 0.0917, Val Acc: 0.9694              
        Epoch 3/6 | Train Loss: 0.0997, Train Acc: 0.9710 | Val Loss: 0.0862, Val Acc: 0.9712              
        Epoch 4/6 | Train Loss: 0.0898, Train Acc: 0.9743 | Val Loss: 0.0742, Val Acc: 0.9763              
        Epoch 5/6 | Train Loss: 0.0834, Train Acc: 0.9761 | Val Loss: 0.0686, Val Acc: 0.9776              
        Epoch 6/6 | Train Loss: 0.0800, Train Acc: 0.9771 | Val Loss: 0.0648, Val Acc: 0.9794              

        ✅ All datasets trained successfully!

    ======================================================================



    multiple layer of parale +sequencial  convolution layer
    kernal 3 + 5 + 7 -> kernal 3 + 5 + 7 -> lstm

    ___________🧩 Using 50% of training data___________
    Epoch 1/6 | Train Loss: 0.2292, Train Acc: 0.9211 | Val Loss: 0.1166, Val Acc: 0.9638              
    Epoch 2/6 | Train Loss: 0.1259, Train Acc: 0.9626 | Val Loss: 0.0915, Val Acc: 0.9704              
    Epoch 3/6 | Train Loss: 0.1064, Train Acc: 0.9671 | Val Loss: 0.0840, Val Acc: 0.9719              
    Epoch 4/6 | Train Loss: 0.0976, Train Acc: 0.9705 | Val Loss: 0.0816, Val Acc: 0.9738              
    Epoch 5/6 | Train Loss: 0.0892, Train Acc: 0.9728 | Val Loss: 0.0726, Val Acc: 0.9784              
    Epoch 6/6 | Train Loss: 0.0843, Train Acc: 0.9742 | Val Loss: 0.0706, Val Acc: 0.9782              
    __________🧩 Using 100% of training data___________
    Epoch 1/6 | Train Loss: 0.2107, Train Acc: 0.9275 | Val Loss: 0.1396, Val Acc: 0.9570              
    Epoch 2/6 | Train Loss: 0.1213, Train Acc: 0.9637 | Val Loss: 0.0839, Val Acc: 0.9711              
    Epoch 3/6 |  Batch 2370/3264 |  Loss: 0.1125, Acc: 0.9688



fully connected layers(dense layers)

lstm layers
50% dataset :- Epoch 6/6 | Train Loss: 0.1011, Train Acc: 0.9709 | Val Loss: 0.0864, Val Acc: 0.9740
100% dataset :-Epoch 6/6 | Train Loss: 0.0862, Train Acc: 0.9732 | Val Loss: 0.0781, Val Acc: 0.9748


using dt model on extracted features of cnn_lstm model

    Xgboost (128 input parms)
        Accuracy: 0.9846410172932187
        Precision: 0.9657971641952592
        Recall: 0.9483473270273143
        F1: 0.956992706992707
        ROC AUC: 0.9704827646162547
    random forest (128 input parms)
        Accuracy: 0.9848325258057721
        Precision: 0.9665403356794803
        Recall: 0.9486661706876395
        F1: 0.957519845526711
        ROC AUC: 0.9707239468650307

using xgboost on handcrafted features + cnn_lstm on url token based features and combinig both using diffrent models on predicted probablity of both 

    using logistic regression(xgboost + cnn_lstm)
        📘 TRAIN METRICS
        Accuracy : 0.9845929926364272
        Precision: 0.9956221828602698
        Recall   : 0.9185321978503767
        F1-score : 0.9555248427890263
        ROC AUC  : 0.9987881565058316

        📗 VALIDATION METRICS
        Accuracy : 0.9793745331980006
        Precision: 0.9922022684310019
        Recall   : 0.8925496864704007
        F1-score : 0.939741509539529
        ROC AUC  : 0.9960990909211707

    using simple mlp(xgboost + cnn_lstm)
        📘 TRAIN METRICS
        Accuracy : 0.9903909683720663
        Precision: 0.9791669468615927
        Recall   : 0.9672507938195007
        F1-score : 0.9731723944339736
        ROC AUC  : 0.9987848469855406

        📗 VALIDATION METRICS
        Accuracy : 0.9847367715494953
        Precision: 0.9668256721595837
        Recall   : 0.9478159209267722
        F1-score : 0.9572264262330274
        ROC AUC  : 0.9960148936912712

    using xgboost(xgboost + cnn_lstm)
        📘 TRAIN METRICS
        Accuracy : 0.9845929926364272
        Precision: 0.9956221828602698
        Recall   : 0.9185321978503767
        F1-score : 0.9555248427890263
        ROC AUC  : 0.9987881565058316

        📗 VALIDATION METRICS
        Accuracy : 0.9793745331980006
        Precision: 0.9922022684310019
        Recall   : 0.8925496864704007
        F1-score : 0.939741509539529
        ROC AUC  : 0.9960990909211707




lstm layers:- 
    128 size bilstm layers
    Epoch 1/2 Training CNN + transformer| Train Loss: 0.0836, Train Acc: 0.9743 | Val Loss: 0.0823, Val Acc: 0.9735
    Epoch 2/2 Training CNN + transformer| Train Loss: 0.0591, Train Acc: 0.9812 | Val Loss: 0.0601, Val Acc: 0.9804
    Epoch 1/2 Training CNN + transformer| Train Loss: 0.0474, Train Acc: 0.9849 | Val Loss: 0.0526, Val Acc: 0.9834
    Epoch 2/2 Training CNN + transformer| Train Loss: 0.0387, Train Acc: 0.9875 | Val Loss: 0.0492, Val Acc: 0.9842
    Epoch 1/1 Training CNN + transformer| Train Loss: 0.0344, Train Acc: 0.9891 | Val Loss: 0.0465, Val Acc: 0.9853

optimizers:- 
    nadams 3,5,7 cnn + 128 lstm:- 
    ======================================================================
    🚀 Training model on DATASET 1 (MALICIOUS URLS) dataset
    ======================================================================
    __________🧩 Using 100% of training data___________
    Epoch 1/2 Training CNN + transformer| Train Loss: 0.0795, Train Acc: 0.9765 | Val Loss: 0.0786, Val Acc: 0.9755
    Epoch 2/2 Training CNN + transformer| Train Loss: 0.0589, Train Acc: 0.9815 | Val Loss: 0.0595, Val Acc: 0.9806
    Epoch 1/2 Training CNN + transformer| Train Loss: 0.0510, Train Acc: 0.9830 | Val Loss: 0.0589, Val Acc: 0.9815
    Epoch 2/2 Training CNN + transformer| Train Loss: 0.0421, Train Acc: 0.9868 | Val Loss: 0.0508, Val Acc: 0.9840
    Epoch 1/1 Training CNN + transformer| Train Loss: 0.0351, Train Acc: 0.9885 | Val Loss: 0.0483, Val Acc: 0.9848

    nadams 3,5,7 cnn + 128 bilstm:- 

    ======================================================================
    🚀 Training model on DATASET 1 (MALICIOUS URLS) dataset
    ======================================================================
    __________🧩 Using 100% of training data___________
    Epoch 1/2 Training CNN + transformer| Train Loss: 0.0852, Train Acc: 0.9732 | Val Loss: 0.0820, Val Acc: 0.9734
    Epoch 2/2 Training CNN + transformer| Train Loss: 0.0672, Train Acc: 0.9797 | Val Loss: 0.0661, Val Acc: 0.9793
    Epoch 1/2 Training CNN + transformer| Train Loss: 0.0470, Train Acc: 0.9849 | Val Loss: 0.0534, Val Acc: 0.9827
    Epoch 2/2 Training CNN + transformer| Train Loss: 0.0404, Train Acc: 0.9867 | Val Loss: 0.0504, Val Acc: 0.9843
    Epoch 1/1 Training CNN + transformer| Train Loss: 0.0353, Train Acc: 0.9882 | Val Loss: 0.0504, Val Acc: 0.9842
    


parrale model 
    with 2 layers
        Epoch 1/2 Training CNN + transformer| Train Loss: 0.1051, Train Acc: 0.9705 | Val Loss: 0.0885, Val Acc: 0.9717
        Epoch 2/2 Training CNN + transformer| Train Loss: 0.0871, Train Acc: 0.9741 | Val Loss: 0.0767, Val Acc: 0.9746
        Epoch 1/2 Training CNN + transformer| Train Loss: 0.0672, Train Acc: 0.9789 | Val Loss: 0.0647, Val Acc: 0.9791
        Epoch 2/2 Training CNN + transformer| Train Loss: 0.0555, Train Acc: 0.9836 | Val Loss: 0.0571, Val Acc: 0.9840
        Epoch 1/1 Training CNN + transformer| Train Loss: 0.0527, Train Acc: 0.9840 | Val Loss: 0.0567, Val Acc: 0.9831
activation layers

pooling layers

normalization layers


