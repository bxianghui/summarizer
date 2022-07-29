from others.tokenization import BertTokenizer
from multiprocess import Pool
from others.logging import logger
import json
import os
import glob
import torch
import gc
from os.path import join as pjoin

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


    def preprocess(self, src, tgt, use_bert_basic_tokenizer=False, is_test=False):

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
        tgt = tgt[:self.args.max_tgt_ntokens]
        tgt_subtoken = self.tokenizer.tokenize(tgt, True)
        # print(tgt_subtoken)
        tgt_subtoken = [self.tgt_bos] + tgt_subtoken + [self.tgt_eos]
        # tgt_subtokens_str = '[unused1] '.join(
        #     [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
        #      in tgt]) + ' [unused2]'
        tgt_subtoken = tgt_subtoken[:self.args.max_tgt_ntokens]
        # print(tgt_subtoken)
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
        # print(tgt_subtoken_idxs)
        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        # return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt
        return src_subtoken_idxs, tgt_subtoken_idxs, segments_ids, src_txt, tgt_txt

def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        # print(corpus_type)
        # for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
        # print(glob.glob(pjoin(args.raw_path, corpus_type)))
        json_f = args.raw_path + '/' + corpus_type
        # for json_f in glob.glob(pjoin(args.raw_path, corpus_type)):
            # print(json_f)
        real_name = json_f.split('\\')[-1]
        src_path = real_name + '.json'
            # print(real_name)
        a_lst = (corpus_type, src_path, args, args.save_path + '/' + real_name.split('/')[-1] + '.bert.pt')
        # print(a_lst)
        _format_to_bert(a_lst)
        # pool = Pool(args.n_cpus)
        # for d in pool.imap(_format_to_bert, a_lst):
        #     pass
        #
        # pool.close()
        # pool.join()

def _format_to_bert(params):
    corpus_type, data_file, args, save_file = params
    print(data_file)
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)
    # print('bert_Data')
    logger.info('Processing %s' % data_file)
    jobs = json.load(open(data_file, encoding='utf-8'))
    # src = open(src_path, 'r', encoding='utf-8')

    datasets = []
    count = 0
    for d in jobs:
        source = d['content']
        tgt = d['title']
        # print(source)
        # print(tgt)
        # print('source:', source)
        # print('tgt:', tgt)
        # sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 1)
        # print('sent_labels:', sent_labels)
        # if (args.lower):
        #     source = [' '.join(s).lower().split() for s in source]
        #     tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
        # print('b_data:', b_data)
        if (b_data is None):
            continue
        src_subtoken_idxs, tgt_subtoken_idxs, segments_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs, "segs": segments_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        # print("src_txt:", src_txt)
        # print('tgt_txt:', tgt_txt)
        datasets.append(b_data_dict)
        if len(datasets) > args.shard_size:
            print(len(datasets))
            # print("datasets:", datasets)
            # print(datasets)
            file = 'lcsts_data/' + corpus_type
            save_file =file + '_' + str(count) + '.bert.pt'
            count += 1
            logger.info('Processed instances %d' % len(datasets))
            logger.info('Saving to %s' % save_file)
            # print(save_file)
            torch.save(datasets, save_file)
            datasets = []
    # src.close()
    # tgt.close()
    if len(datasets) > 0:
        print(len(datasets))
        # print("datasets:", datasets)
    # print(datasets)
        file = 'lcsts_data/' + corpus_type
        save_file =file + '_' + str(count) + '.bert.pt'
        count += 1
        logger.info('Processed instances %d' % len(datasets))
        logger.info('Saving to %s' % save_file)
    # print(save_file)
        torch.save(datasets, save_file)

        datasets = []
        gc.collect()

# def cut_sent(infile, outfile):
#   cutLineFlag = ["？", "！", "。","…"] #本文使用的终结符，可以修改
#   sentenceList = []
#   with open(infile, "r", encoding="UTF-8") as file:
#     oneSentence = ""
#     for line in file:
#       if len(line)!=0:
#         sentenceList.append(line.strip() + "\r")
#         oneSentence=""
#       # oneSentence = ""
#       for word in line:
#         if word not in cutLineFlag:
#           oneSentence = oneSentence + word
#         else:
#           oneSentence = oneSentence + word
#           if oneSentence.__len__() > 4:
#             sentenceList.append(oneSentence.strip() + "\r")
#           oneSentence = ""
#       if len(oneSentence) != 0:
#         sentenceList.append(oneSentence.strip() + "\r")
#         oneSentence = ""
#   with open(outfile, "w", encoding="UTF-8") as resultFile:
#     print(sentenceList.__len__())
#     resultFile.writelines(sentenceList)

if __name__ == '__main__':

    with open('nlpcc/valid.json', 'r', encoding='utf-8') as r:
        res = r.readlines()
        with open('nlpcc_data/valid.json', 'w', encoding='utf-8') as w:
            w.write(res[0][:-2])
            w.write(']')