from tkinter import *
import re
from others.tokenization import BertTokenizer
from models.predictor import build_predictor
from others.logging import logger, init_logger
from models.model_builder import Z_AbsSummarizer
import os
import argparse
import torch
import translate
# 河南公安厅出台举报暴恐犯罪线索奖励5万元,将鼓励公民和组织积极举报该办法将于今年7月1日实施行。
MAX_Z_LENGTH=512


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused1]'
        self.tgt_eos = '[unused2]'
        # self.tgt_sent_split = '[unused3]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]


    def preprocess(self, src, is_test=True):

        if ((not is_test) and len(src) == 0):
            return None
        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        # print(idxs)

        _sent_labels = [0] * len(src)

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]
        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None
        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        # print(src)
        src_subtokens = self.tokenizer.tokenize(text, True)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        # print(src_subtokens)
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        # print(src_subtoken_idxs)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        # cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        # sent_labels = sent_labels[:len(cls_ids)]

        # tgt_subtokens_str = '[unused1] ' + ' [unused3] '.join(
        #     [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
        #      in tgt]) + ' [unused2]'
        src_txt = [original_src_txt[i] for i in idxs]

        # return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt
        return src_subtoken_idxs, segments_ids, src_txt

def format_bert(x):
    content = clear_character(x)
    content = clear_space(content)
    sen_list = cut_sent(content)
    deal_list = []
    for sen in sen_list:
        cont = clear_character(sen)
        deal_list.append(cont)
    content = deal_list
    return content

def clear_space(text):
    content1 = re.sub("[[0-9]{8,20}", '', text)
    content2 = re.sub("[[a-zA-Z]{8,20}", '', content1)
    content3 = content2.replace(' ', '')  # 去掉文本中的空格
    content4 = content3.replace('...', ',')
    # content3 = re.sub('\d+', '', content2)

    return content4
def clear_character(sentence):
    pattern = re.compile("[^\u4e00-\u9fa5^,^，^。^.^!^a-z^A-Z^0-9]")  #只保留中英文、数字和符号，去掉其他东西
    #若只保留中英文和数字，则替换为[^\u4e00-\u9fa5^a-z^A-Z^0-9]
    line=re.sub(pattern,'',sentence)  #把文本中匹配到的字符替换成空字符
    new_sentence = ''.join(line.split())    #去除空白
    return new_sentence

def cut_sent(text):
    cutLineFlag = ["？", "！", "。","…"] #本文使用的终结符，可以修改
    sentenceList = []
    oneSentence=""
    for word in text:
        if word not in cutLineFlag:
            oneSentence = oneSentence + word
        else:
            oneSentence = oneSentence + word
            if oneSentence.__len__() > 4:
                sentenceList.append(oneSentence)
                oneSentence = ""
    if len(oneSentence) != 0:
        sentenceList.append(oneSentence)
        oneSentence = ""
    return sentenceList

def _pad(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data

def process(ex, is_text, args):
    src = ex['src']
    segs = ex['segs']
    if (not args.use_interval):
        segs = [0] * len(segs)
    src_txt = ex['src_txt']

    if 'z' in ex:
        z = ex['z']
        if len(z) == 0:
            z = [63]
            z_segs = [0]
        else:
            z_segs = [0] * len(z)
            end_id = [z[-1]]
            z = z[:-1][:MAX_Z_LENGTH - 1] + end_id
            z_segs = z_segs[:MAX_Z_LENGTH]
    else:
        z = [63]
        z_segs = [0]
    end_id = [src[-1]]
    src = src[:-1][:args.max_pos - 1] + end_id
    segs = segs[:args.max_pos]
    src_list = []
    src_list.append(src)
    segs_list = []
    segs_list.append(segs)
    src = torch.tensor(_pad(src_list, 0)).cuda()
    segs = torch.tensor(_pad(segs_list, 0)).cuda()
    mask_src = 1 - (src == 0)
    z_list = []
    z_list.append(z)
    z_segs_list = []
    z_segs_list.append(z_segs)

    z = torch.tensor(_pad(z_list, 0)).cuda()
    z_segs = torch.tensor(_pad(z_segs_list, 0)).cuda()
    mask_z = 1 - (z == 0)
    return src, segs, mask_src, z, z_segs, mask_z, src_txt

def translate(args, pt, step, x):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    x = format_bert(x)
    print(x)
    bert = BertData(args)
    batch = {}
    src_subtoken_idxs, segments_ids, src_txt = bert.preprocess(x)
    batch['src'] = src_subtoken_idxs
    batch['segs'] = segments_ids
    batch['src_txt'] = src_txt
    src, segs, mask_src, z, z_segs, mask_z, src_txt = process(batch, True, args)

    batch['src'] = src
    batch['segs'] = segs
    batch['mask_src'] = mask_src
    batch['src_txt'] = src_txt
    batch['z'] = z
    batch['mask_z'] = mask_z
    batch['z_segs'] = z_segs
    checkpoint = torch.load('models_path/model_step_200000.pt', map_location=lambda storage, loc: storage)
    model = Z_AbsSummarizer(args, device, checkpoint)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
               'PAD': tokenizer.vocab['[PAD]']}
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    batch_ite = []
    batch_ite.append(batch)
    result = predictor.translate(batch_ite, step)
    return result

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-debug", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test', 'oracle'])
    parser.add_argument("-bert_data_path", default='../bert_data_new/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument('-min_src_nsents', default=5, type=int)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    # parser.add_argument('-min_src_nsents', default=2, type=int)
    # parser.add_argument('-max_src_nsents', default=20, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=10, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=150, type=int)
    # parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
    # parser.add_argument('-max_src_ntokens_per_sent', default=50, type=int)
    parser.add_argument('-min_tgt_ntokens', default=20, type=int)
    parser.add_argument('-max_tgt_ntokens', default=128, type=int)
    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-test_batch_size", default=1, type=int)

    parser.add_argument("-max_pos", default=1024, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=256, type=int)
    parser.add_argument("-enc_ff_size", default=256, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha", default=0.95, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=60, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    parser.add_argument("-copy", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='./logs/cnndm.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='models_path/model_step_200000.pt')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1
    cp = args.test_from
    try:
        step = int(cp.split('.')[-2].split('_')[-1])
    except:
        step = 0
    txt = txtone.get("0.0",'end')
    with open("test.txt","w",encoding = 'utf-8') as f :
        f.write(txt)
    with open("test.txt", "r", encoding='utf-8') as njuptcs:
        txt = [line.strip() for line in open("test.txt", "r",encoding='utf-8').readlines()]
        text = njuptcs.read().replace('\n', '')
    print ('原文为：')
    for i in txt:
        print (i)
    summarize_text =  translate(args, cp, step, text)
    print(summarize_text)
    #summarize_text = textrank.summarize(text,5)
    #ing = 0
    for i in summarize_text:
        #ing += 1
        #txttwo.insert('end',ing)
        #txttwo.insert('end','、')
        txttwo.insert('end',i)
        #print (ing,i)
    txttwo.insert('end','\n')
myWindow = Tk()
myWindow.minsize(830,480)
myWindow.title("自动摘要生成方法")
Label(myWindow, text = "输入文本",font = ('微软雅黑 12 bold')).grid(row = 0,column = 0)
Label(myWindow, text = "输出摘要",font = ('微软雅黑 12 bold')).grid(row = 0,column = 2)
entry1=Entry(myWindow)
txtone = Text(entry1)
txtone.pack()
entry2=Entry(myWindow)
txttwo = Text(entry2)
txttwo.pack()
entry1.grid(row = 0, column = 1)
entry2.grid(row = 0, column = 3)
#Quit按钮退出；Run按钮打印计算结果
Button(myWindow, text='退出',command = myWindow.quit, font = ('微软雅黑 12 bold')).grid(row = 1, column = 3,sticky = S,padx = 20,pady = 20)
Button(myWindow, text='生成摘要',command = run, font = ('微软雅黑 12 bold')).grid(row = 1, column = 1, sticky = S, padx = 20,pady = 20)
#进入消息循环
myWindow.mainloop()
