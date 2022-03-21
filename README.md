# ml-for-healthcare

### task 1

vanilla cnn results on mitbih\
accuracy: 0.9787

vanilla lstm results on mitbih\
accuracy: 0.9775

vanilla cnn results on ptbdb\
accuracy: 0.9742 - auroc: 0.9935 - auprc: 0.9965

vanilla lstm results on ptbdb\
accuracy: 0.7221 - auroc: 0.5005 - auprc: 0.7236

### task 2

bidirectional lstm on mitbih\
accuracy: 0.9863

simple_rnn attention on mitbih\
accuracy: 0.9255

lstm_attention on mitbih\
accuracy: 0.9847

inception_net on mitbih\
accuracy: 0.9848

bidirectional lstm on ptbdb\
accuracy: 0.9742 - auroc: 0.9927 - auprc: 0.9955

simple_rnn attention on ptbdb\
accuracy: 0.7667 - auroc: 0.8287 - auprc: 0.9283

lstm_attention on ptbdb\
accuracy: 0.7513 - auroc: 0.8140 - auprc: 0.9245

inception_net on ptbdb\
accuracy: 0.9821 - auroc: 0.9945 - auprc: 0.9965


### task 4

transfer learning train all layers on ptbdb\
accuracy: 0.9907 - auroc: 0.9960 - auprc: 0.9972

transfer learning freeze lstm on ptbdb\
accuracy: 0.9804 - auroc: 0.9930 - auprc: 0.9955
