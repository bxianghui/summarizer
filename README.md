# summarizer
## 中文文本摘要生成模型

# 模型：
![image](https://user-images.githubusercontent.com/90383015/181683093-8389baf0-4017-4bf8-bd29-fc70ea011e30.png)

训练命令：
python z_train.py  -task abs -mode train -bert_data_path data/lcsts -dec_dropout 0.2  -model_path models_path -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 4 -train_steps 100000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 128 -visible_gpus 0  -log_file ../logs/pre_data_cnndm
验证：
python z_train.py -task abs -mode validate -batch_size 3000 -test_batch_size 1500 -bert_data_path bert_data -log_file logs/test.logs -model_path nlpcc_models_path -sep_optim true -use_interval true -visible_gpus 0 -max_pos 1024 -max_length 128 -alpha 0.95 -min_length 15 -result_path logs/nlpcc_validate
测试命令：
python -W ignore z_train.py -task abs -mode test -batch_size 3000 -test_batch_size 1500 -bert_data_path bert_data -log_file logs/test.logs -test_from C:\Users\bu\Desktop\models_path -sep_optim true -use_interval true -visible_gpus 0 -max_pos 1024 -max_length 128 -alpha 0.95 -min_length 15 -result_path nlpcc1_result


使用可视化界面：
![自动摘要生成方法图5](https://user-images.githubusercontent.com/90383015/181683497-68fc6479-b159-4047-9934-5c0ee60a0262.jpg)


