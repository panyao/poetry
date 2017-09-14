# poetry
Generating classical Chinese poetry with RNN

Use Recurrent Neural Network and LSTM to generate classical Chinese peotry.

##
Some generted samples:

˭ν��ʱ��������˭�ʷ硣
�ƽ�Ϊ���ѣ����������顣
����һ��ˮ�����ٻ�����
���������У�����Ī��͡�
��ȳԶ��ƽɳȥδ�١�
���������£�����ȥ������

�����������ɫ���㡣
�翪��������������֦����
���������գ�����δ���
���������ʣ���Ӱ�����衣
�����˿ϸ�����������ߡ�
������ɫԶ����ˮ����¥��

��ľ����Ͼ���ֵ������
���鳯���������к�����
�����α�����������ʪ��
��������䣬����������
��������ã����������



## Running instructions
 - Put training peotry text file under 'data' directory.
 - Run preprocess.py
 - Run train_rnn.py to train the model.
 - Run generate.py to generate poetry with given first word.