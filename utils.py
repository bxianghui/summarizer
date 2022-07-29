import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import config
import rouge

def load_vocab(vocabulary_file, data_file):
    if os.path.exists(vocabulary_file):
        chars, id2char, char2id = json.load(open(vocabulary_file))
        id2char = {int(i):j for i, j ,in id2char.items()}
    else:
        print('No vocabulary found in path {}, constructing new vocab file'.format(vocabulary_file))
        df = pd.read_csv(data_file)
        chars = {}
        pd.set_trace()
        for text, summarization in tqdm(df.values):
            for w in text:
                chars[w] = chars.get(w, 0) + 1
            for w in summarization:
                chars[w] = chars.get(w, 0) + 1
        chars = {i: j for i, j in chars.items() if j >= config.min_count}
        # 0: padding
        # 1: unk
        # 2: start
        # 3: end
        id2char = {i + 4: j for i, j in enumerate(chars)}
        char2id = {j: i for i, j in id2char.items()}
        json.dump([chars, id2char, char2id], open(vocabulary_file, 'w'))
        
    return chars, id2char, char2id


def str2id(strings, char2id, start_end = False):
    if start_end:  # <start>+summary+<end> for summary
        ids = [char2id.get(c, 1) for c in strings[:config.max_len - 2]]
        ids = [2] + ids + [3]
    else:  # for text
        ids = [char2id.get(c, 1) for c in strings[:config.max_len]]
    return ids

def id2str(ids, id2char):
    return ''.join([id2char.get(i, '') for i in ids])



def beam_search(s, model, device, char2id, id2char, beam_width = 3, max_len = 64):
    xid = np.array([str2id(s, char2id)] * beam_width)
    yid = np.array([[2]] * beam_width)
    scores = [0] * beam_width
    len_s = len(xid[0])

    for i in range(max_len):
        model.eval()
        with torch.no_grad():
            x, y = torch.tensor(xid).to(device), torch.tensor(yid).to(device)
            len_x, len_y = torch.tensor([len_s] * beam_width), torch.tensor([i+1] * beam_width)
            prob = model(x, y, len_x, len_y)[:, i, 3:].clone().cpu().numpy()
        #log_prob = np.log(prob + 1e-6)
        log_prob = prob
        arg_topk = log_prob.argsort(axis = 1)[:, -beam_width:]

        tmp_yid = []
        tmp_scores = []
        if i == 0:
            for j in range(beam_width):
                tmp_yid.append(list(yid[j]) + [arg_topk[0][j] + 3])
                tmp_scores.append(scores[j] + log_prob[0][arg_topk[0][j]])
        else:
            for j in range(beam_width):
                for k in range(beam_width):
                    tmp_yid.append(list(yid[j]) + [arg_topk[j][k] + 3])
                    tmp_scores.append(scores[j] + log_prob[j][arg_topk[j][k]])
                    
            tmp_arg_topk = np.argsort(tmp_scores)[-beam_width:]
            tmp_yid = [tmp_yid[k] for k in tmp_arg_topk]
            tmp_scores = [tmp_scores[k] for k in tmp_arg_topk]

        yid = np.array(tmp_yid)
        scores = np.array(tmp_scores)

        ends = np.where(yid[:, -1] == 3)[0]
        if len(ends) > 0:
            index = ends[scores[ends].argmax()]
            return id2str(yid[index], id2char)

    return id2str(yid[np.argmax(scores)], id2char)



def evaluate(data, model, device, char2id, id2char):
    
    pred_summary = []
    true_summary = []
    for dd in tqdm(data, desc = 'Evaluating:'):
        text, summary = dd
        text = text[0]
        summary = summary[0]
        pred_summary.append(beam_search(text, model, device, char2id, id2char))
        true_summary.append(summary)
        
    rouge_1 = rouge.Rouge().get_scores(pred_summary, true_summary)[0]['rouge-1']['f']
    rouge_2 = rouge.Rouge().get_scores(pred_summary, true_summary)[0]['rouge-2']['f']
    rouge_l = rouge.Rouge().get_scores(pred_summary, true_summary)[0]['rouge-l']['f']

    print('Rouge-1 score:', rouge_1)
    print('Rouge-2 score:', rouge_2)
    print('Rouge-l score:', rouge_l)
    
    
    # print('Sample sentences:')
    # sample_1 = '四海网讯，近日，有媒体报道称：章子怡真怀孕了!报道还援引知情人士消息称，“章子怡怀孕大概四五个月，预产期是年底前后，现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息，23日晚8时30分，华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士，这位人士向华西都市报记者证实说：“子怡这次确实怀孕了。她已经36岁了，也该怀孕了。章子怡怀上汪峰的孩子后，子怡的父母亲十分高兴。子怡的母亲，已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时，华西都市报记者为了求证章子怡怀孕消息，又电话联系章子怡的亲哥哥章子男，但电话通了，一直没有人接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和汪峰恋情以来，就被传N遍了!不过，时间跨入2015年，事情却发生着微妙的变化。2015年3月21日，章子怡担任制片人的电影《从天儿降》开机，在开机发布会上几张合影，让网友又燃起了好奇心：“章子怡真的怀孕了吗?”但后据证实，章子怡的“大肚照”只是影片宣传的噱头。过了四个月的7月22日，《太平轮》新一轮宣传，章子怡又被发现状态不佳，不时深呼吸，不自觉想捂住肚子，又觉得不妥。然后在8月的一天，章子怡和朋友吃饭，在酒店门口被风行工作室拍到了，疑似有孕在身!今年7月11日，汪峰本来在上海要举行演唱会，后来因为台风“灿鸿”取消了。而消息人士称，汪峰原来打算在演唱会上当着章子怡的面宣布重大消息，而且章子怡已经赴上海准备参加演唱会了，怎知遇到台风，只好延期，相信9月26日的演唱会应该还会有惊喜大白天下吧。'
    # sample_2 = '8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
    #
    # print(beam_search(sample_1, model, device, char2id, id2char))
    # print(beam_search(sample_2, model, device, char2id, id2char))
