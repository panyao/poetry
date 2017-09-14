# poetry
Generating classical Chinese poetry with RNN

Use Recurrent Neural Network and LSTM to generate classical Chinese peotry.

##
Some generted samples:

谁谓此时绝，春来谁问风。
黄金为旧友，风雨早来归。
旧隐一心水，秋风百花长。
独怀当得行，无以莫相和。
风度愁长远，平沙去未迟。
中宵东作侣，日日去依依。

春雨初开，柳色垂香。
风开兰扇满，柳拂碧枝垂。
玉树无清日，风轻未觉深。
云鬟轻气彩，树影度帘疏。
花落红丝细，池塘碧树催。
高吟春色远，流水绕青楼。

古木心稍暇，贾岛还光辉
告篱朝积道，拂托忽断意
告引宋遍来，含滋欲湿花
秋湖意尚冷，愁掷掷空沈
告性曾令好，清光两何如



## Running instructions
 - Put training peotry text file under 'data' directory.
 - Run preprocess.py
 - Run train_rnn.py to train the model.
 - Run generate.py to generate poetry with given first word.