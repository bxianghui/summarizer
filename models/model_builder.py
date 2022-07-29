import copy

import math

import jieba
import jieba.posseg as psg
from jieba import analyse
import functools
import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_


from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
#
# from models.decoder import TransformerDecoder, TransformerDecoderLayer
from models.decoder import TransformerDecoder, TransformerDecoderLayer
from models.encoder import Classifier, ExtTransformerEncoder, TransformerEncoderLayer, AbsTransformerEncoder
from models.optimizers import Optimizer


class Attention(nn.Module):
    def __init__(self,
                 num_attention_heads,
                 hidden_size):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # usually hidden_size = all_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

    def mask(self, x, mask=None, mode='mul'):
        if mask == None:
            print('Hint: mask is None, original x is returned.')
            return x

        # x shape = [batch_size, seq_len, any_size]
        # mask shape = [batch_size, seq_len, 1]
        if len(x.shape) > 3:  # if x shape = [batch_size, seq_len, size_1, size_2]
            original_shape = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1)  # reshape to [batch_size, seq_len, size_1 * size_2]

        # [batch_size, seq_len, 1] -> [batch_size, seq_len, any_size]
        mask = mask.repeat(1, 1, x.shape[-1])

        if mode == 'mul':
            x = x * mask
        elif mode == 'add':
            x = x - (1 - mask) * 1e10
        else:
            raise ValueError('Got mode {}, Only accept mode to be "add" or "mul"'.format(mode))

        if x.shape != original_shape:  # view back to [batch_size, seq_len, size_1, size_2]
            x = x.reshape(*original_shape)
        return x

    def do_transpose(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # [batch_size, seq_len, out_dim] -> [batch_size, seq_len, num_heads, head_size]
        x = x.reshape(*new_shape)
        # [batch_size, seq_len, num_heads, head_size] -> [batch_size, num_heads, seq_len, head_size]
        return x.permute(0, 2, 1, 3)

    def forward(self, qx, kx, vx, v_mask=None, q_mask=None):
        # we assume kx == vx, so seq_len_k = seq_len_v
        # if mask is passed, size should be [batch_size, seq_len, 1]

        # [batch_size, seq_len_q/kv, hidden_size] -> [batch_size, seq_len_q/kv, all_head_size]
        qw = self.query(qx)
        kw = self.key(kx)
        vw = self.value(vx)

        # [batch_size, num_heads, seq_len_q/kv, head_size]
        qw = self.do_transpose(qw)
        kw = self.do_transpose(kw)
        vw = self.do_transpose(vw)

        # [batch_size, num_heads, seq_len_q, seq_len_kv]
        attention_scores = torch.matmul(qw, kw.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if v_mask is not None:
            # [batch_size, seq_len_kv, seq_len_q, num_heads]
            attention_scores = attention_scores.permute(0, 3, 2, 1)
            attention_scores = self.mask(attention_scores, v_mask, mode='add')
            # [batch_size, num_heads, seq_len_q, seq_len_kv]
            attention_scores = attention_scores.permute(0, 3, 2, 1)

        # do softmax on seq_len_k, because we need to find weights of key sequence
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # [batch_size, num_heads, seq_len_q, seq_len_kv] -> [batch_size, num_heads, seq_len_q, head_size]
        output = torch.matmul(attention_probs, vw)
        # [batch_size, seq_len_q, num_heads, head_size]
        output = output.permute(0, 2, 1, 3)

        # [batch_size, seq_len_q, hidden_size]
        output_shape = output.size()[:-2] + (self.all_head_size,)
        output = output.reshape(*output_shape)

        if q_mask is not None:
            output = self.mask(output, q_mask, mode='mul')

        # [batch_size, seq_len_q, hidden_size]
        return output


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    return optim


def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def get_copy_generator(dec_hidden_size, device):
    gen_func = nn.Sigmoid()
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size * 3, 1),
        gen_func
    )
    generator.to(device)

    return generator


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        # if(large):
        #     self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        # else:
        self.model = BertModel.from_pretrained('bert-base-chinese', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if (self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads,
                                     intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if (args.max_pos > 512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls


class Z_AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(Z_AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if (args.max_pos > 512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings,
            vocab_size=self.vocab_size)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight

        self.f1 = TransformerEncoderLayer(self.bert.model.config.hidden_size, args.ext_heads, args.ext_ff_size,
                                          args.ext_dropout)
        self.f2 = TransformerEncoderLayer(self.bert.model.config.hidden_size, args.ext_heads, args.ext_ff_size,
                                          args.ext_dropout)
        self.keywords_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        self.keywords_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
        self.encoder = AbsTransformerEncoder(self.bert.model.config.hidden_size, args.ext_heads, args.ext_ff_size,
                                             args.ext_dropout)
        self.attention = Attention(args.ext_heads, self.bert.model.config.hidden_size)
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if (args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, mask_src, mask_tgt, z, mask_z, z_segs, src_text):
        top_vec = self.bert(src, segs, mask_src)
        # word_embedding = top_vec.cpu().detach().numpy()
        # matlib(word_embedding, "attn.pdf")
        top_vec = self.f1(1, top_vec, top_vec, 1 - mask_src)
        # word_embedding = top_vec.cpu().detach().numpy()
        # matlib(word_embedding, "word.pdf")
        inputs, seq = self.get_key_words_embedding(self.args, src, src_text, top_vec)
        # print(inputs.shape)
        inputs_mask = self.get_pad_mask(seq, pad_idx=0)
        # print(inputs_mask)
        enc_output = self.encoder(inputs, 1 - inputs_mask)
        # print(enc_output.shape)
        attn_output = self.attention(enc_output, top_vec, top_vec)
        # attn_embedding = attn_output.cpu().detach().numpy()
        # print(attn_output.shape)
        dec_state = self.decoder.init_decoder_state(src, top_vec, seq, attn_output)
        # print(dec_state.shape)
        decoder_outputs, state, copy_prob = self.decoder(tgt[:, :-1], top_vec, attn_output, dec_state)
        # z_top_vec = self.bert(z, z_segs, mask_z)
        # print(z.shape)
        # print(z_top_vec.shape)
        # z_top_vec = self.f2(1, z_top_vec, z_top_vec, 1-mask_z)
        # print(z_top_vec.shape)
        # dec_state = self.decoder.init_decoder_state(src, top_vec, z, z_top_vec)
        # print(dec_state.shape)
        # decoder_outputs, state, copy_prob = self.decoder(tgt[:, :-1], top_vec, z_top_vec, dec_state)

        return decoder_outputs, None, [copy_prob[0], copy_prob[1], copy_prob[2], enc_output[:, 0, :]]

    def get_key_words_embedding(self, args, src, src_txt, sen_embed):
        # print(src_txt)
        batch_src, p_src, dim_src = sen_embed.size()
        text = ''
        src_text = ''
        for i, item in enumerate(src_txt[0]):
            item = item.replace(' ', '')
            src_text = src_text + ' ' + item + ' '
            text = text + item
        text = text[:1024]
        keywords_flag = textrank_extract(text)
        # print(keywords_flag)
        # print(len(src_text))
        keywords_list = []
        for item in keywords_flag:
            word, length = item
            # print(text.index(word))
            pos = index_word(word, src_text)
            keywords_list.append((word, pos, length))

        index_list = []  # 存储关键字所在的索引
        for item in keywords_list:
            word, pos, length = item
            count = 0
            for _ in range(length):
                index_list.append(pos + count)
                count += 1
                # print(src_text[pos])
        # print(keywords_list)
        # print(index_list)
        index_list.sort()
        key_tensors = []
        # print(index_list)
        # print(sen_embed.shape)
        batch, pos, dim = sen_embed.size()
        for i in index_list:
            if i < args.max_pos:
                if i < pos:
                    key_tensors.append(sen_embed[0][i])
        # key_tensors = torch.tensor(key_tensors)
        if len(key_tensors) == 0:
            return self.keywords_embeddings(torch.zeros(batch_src, p_src).long().cuda()), torch.zeros(batch_src, p_src).long().cuda()
        key_tensors = torch.tensor([item.cpu().detach().numpy() for item in key_tensors]).cuda()
        # key_tensors = torch.tensor([item.detach().numpy() for item in key_tensors])
        key_tensors = key_tensors.unsqueeze(0)
        # print(key_tensors)
        # print(key_tensors.shape)
        # print(index_tensor[0])
        # print(index_tensor[0])
        # print(src)
        src_list = src.cpu().detach().numpy().tolist()
        key_l = []
        for i in index_list:
            if i < args.max_pos:
                if i < pos:
                    key_l.append(src_list[0][i])
        key_l_b = [key_l]
        # print(key_l_b)
        batch1, p1, dim = key_tensors.size()
        if len(key_l) == 0:
            key_l_b = torch.zeros(batch1, p1)
        key_l_b = torch.tensor(key_l_b).long().cuda()
        key_embed1 = self.keywords_embeddings(key_l_b)

        # batch2, p2, dim = key_tensors.size()
        # if p1 > p2:
        #     key_embed1 = key_embed1[:, :p2, :]
        # elif p2 > p1:
        #     key_tensors = key_tensors[:, :p1, :]
        # print(key_embed1)
        # print(key_tensors)
        final_embed = (key_tensors + key_embed1) / 2
        return final_embed, key_l_b

    def get_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx)


# 停用词表加载方法
def get_stopword_list():
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stop_word_path = '../stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path, encoding='utf-8').readlines()]
    return stopword_list


# 分词方法，调用结巴接口
def seg_to_list(sentence, pos=False):
    if not pos:
        # 不进行词性标注的分词方法
        seg_list = jieba.cut(sentence)
    else:
        # 进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list


# 去除干扰词
def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []
    # 根据POS参数选择是否词性过滤
    ## 不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        # 过滤停用词表中的词，以及长度为<2的词
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)

    return filter_list


#  排序函数，用于topK关键词的按值排序
def cmp(e1, e2):
    import numpy as np
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


def textrank_extract(text, pos=False, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    keywords_flag = []
    # 输出抽取出的关键词
    for keyword in keywords:
        # print(keyword + "/ ", end='')
        keywords_flag.append((keyword, len(keyword)))
    # print()
    return keywords_flag


def index_word(word, src):
    pos = src.index(word)
    return pos


def matlib(word_embedding, name):
    label = []
    for values in word_embedding[0]:
        label.append(np.argmax(values))

    # 将词向量转化为2维向量
    fea = TSNE(n_components=2).fit_transform(word_embedding[0])
    #
    # fea = PCA(n_components=2).fit_transform(word_embedding[1])
    pdf = PdfPages(name)

    # 画散点图
    # 更多颜色请查看[https://www.cnblogs.com/qianblue/p/10783261.html]
    cValue = ['red', 'yellow', 'green', 'blue', 'orangered', 'steelblue', 'slateblue', 'tomato', 'peru', 'darkorange',
              'deeppink', 'crimson']
    cls = np.unique(label)
    fea_num = [fea[label == i] for i in cls]
    for i, f in enumerate(fea_num):
        if cls[i] in range(10):  # 如果类别标签为10以内的数字，则使用'+'进行标记
            plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='+', edgecolor='none', c=cValue[i])
        else:
            plt.scatter(f[:, 0], f[:, 1], label=cls[i], edgecolor='none', c=cValue[i])

    plt.tight_layout()
    pdf.savefig()
    plt.show()
    pdf.close()
