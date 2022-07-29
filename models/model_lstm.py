import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import math
from models.optimizers import Optimizer
from pytorch_transformers import BertModel, BertConfig
from models.decoder import TransformerDecoder, TransformerDecoderLayer
from models.encoder import TransformerEncoderLayer

class ScaleShift(nn.Module):
   # scale and shift layer 
   def __init__(self, 
                input_shape, 
                init_value, # 1e-3
                device):
       
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value] * input_shape).to(device))
       self.bias = nn.Parameter(torch.FloatTensor([init_value] * input_shape).to(device))

   def forward(self, inputs):
       return torch.exp(self.scale) * inputs + self.bias


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
        
    
    def mask(self, x, mask = None, mode = 'mul'):
        if mask == None:
            print('Hint: mask is None, original x is returned.')
            return x
        
        # x shape = [batch_size, seq_len, any_size]
        # mask shape = [batch_size, seq_len, 1]
        if len(x.shape)>3: # if x shape = [batch_size, seq_len, size_1, size_2]
            original_shape = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1) # reshape to [batch_size, seq_len, size_1 * size_2]
        
        # [batch_size, seq_len, 1] -> [batch_size, seq_len, any_size]
        mask = mask.repeat(1, 1, x.shape[-1])
        
        if mode == 'mul':
            x = x * mask
        elif mode == 'add':
            x = x - (1 - mask) * 1e10
        else:
            raise ValueError('Got mode {}, Only accept mode to be "add" or "mul"'.format(mode))
        
        if x.shape != original_shape: # view back to [batch_size, seq_len, size_1, size_2]
            x = x.reshape(*original_shape)
        return x
        
    def do_transpose(self, x):
       new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
       # [batch_size, seq_len, out_dim] -> [batch_size, seq_len, num_heads, head_size]
       x = x.reshape(*new_shape)
       # [batch_size, seq_len, num_heads, head_size] -> [batch_size, num_heads, seq_len, head_size]
       return x.permute(0, 2, 1, 3)
        
    def forward(self, qx, kx, vx, v_mask = None, q_mask = None):
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
            attention_scores = self.mask(attention_scores, v_mask, mode = 'add')
            # [batch_size, num_heads, seq_len_q, seq_len_kv]
            attention_scores = attention_scores.permute(0, 3, 2, 1)
        
        # do softmax on seq_len_k, because we need to find weights of key sequence
        attention_probs = nn.Softmax(dim = -1)(attention_scores)
        
        # [batch_size, num_heads, seq_len_q, seq_len_kv] -> [batch_size, num_heads, seq_len_q, head_size]
        output = torch.matmul(attention_probs, vw)
        # [batch_size, seq_len_q, num_heads, head_size]
        output = output.permute(0, 2, 1, 3)
        
        # [batch_size, seq_len_q, hidden_size]
        output_shape = output.size()[:-2] + (self.all_head_size, )
        output = output.reshape(*output_shape)
        
        if q_mask is not None:
            output = self.mask(output, q_mask, mode = 'mul')
        
        # [batch_size, seq_len_q, hidden_size]
        return output

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
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


        
class LSTM_Model(nn.Module):
    def __init__(self, 
                 args,
                 device,
                 checkpoint=None,
                 bert_from_extractive=None
                 ):
        
        super().__init__()
        
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

        if (args.max_pos > 1024):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:1024] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[1024:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 1024, 1)
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
        self.f2 = TransformerEncoderLayer(self.bert.model.config.hidden_size, args.ext_heads, args.ext_ff_size, args.ext_dropout)

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
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        # self.embedding = nn.Embedding(num_embeddings = vocab_size + 4,
        #                               embedding_dim = embedding_size)
        # self.lstm_x = nn.LSTM(input_size = embedding_size,
        #                     hidden_size = hidden_size // 2,
        #                     num_layers = 2,
        #                     batch_first = True,
        #                     bidirectional = True)
        #
        # self.lstm_y = nn.LSTM(input_size = embedding_size,
        #                     hidden_size = hidden_size,
        #                     num_layers = 2,
        #                     batch_first = True)
        #
        # self.scale_shift = ScaleShift(input_shape = vocab_size + 4, init_value = 1e-3, device = device)
        #
        # self.layer_norm_x = nn.LayerNorm(normalized_shape = hidden_size)
        # self.layer_norm_y = nn.LayerNorm(normalized_shape = hidden_size)
        # 错
        # self.linear_1 = nn.Linear(in_features =self.bert.model.config.hidden_size, out_features = self.args.enc_hidden_size)
        # self.linear_2 = nn.Linear(in_features = args.dec_hidden_size, out_features = self.vocab_size)
        #
        # self.attention = Attention(num_attention_heads = 8,
        #                            hidden_size = args.dec_hidden_size)
        #
        # self.activation = nn.ReLU(inplace = True)
        # self.softmax = nn.LogSoftmax(dim = -1)
        self.to(device)
        
    
    def to_one_hot(self, x, mask):
        # [batch_size, seq_len] -> [batch_size, seq_len, vocab_size+4]
        x = F.one_hot(x, num_classes = self.vocab_size + 4)
        # [batch_size, seq_len, 1] -> [batch_size, seq_len, vocab_size+4]
        mask = mask.repeat(1, 1, self.vocab_size + 4)
        # [batch_size, 1, vocab_size+4]
        x = torch.sum(x * mask, dim = 1, keepdim = True) 
        x = torch.where(x >= 1., torch.tensor([1.]).to(self.device), torch.tensor([0.]).to(self.device))
        return x
        
    
    def forward(self, src, tgt, segs, mask_src, mask_tgt, z, mask_z, z_segs):
        top_vec = self.bert(src, segs, mask_src)
        top_vec = self.f1(1, top_vec, top_vec, 1 - mask_src)
        print(top_vec.shape)
        z_top_vec = self.bert(z, z_segs, mask_z)
        z_top_vec = self.f2(1, z_top_vec, z_top_vec, 1-mask_z)
        dec_state = self.decoder.init_decoder_state(src, top_vec, z, z_top_vec)
        decoder_outputs, state, copy_prob = self.decoder(tgt[:, :-1], top_vec, z_top_vec, dec_state)

        # x is input text, shape = [batch_size, seq_len_x]
        # y is label summarization, shape = [batch_size, seq_len_y]
       
        # mask = [batch_size, seq_len, 1]
        # x_mask = torch.where(x > 1., torch.tensor([1.]).to(self.device), torch.tensor([0.]).to(self.device)).unsqueeze(-1)
        
        # [batch_size, 1, vocab_size+4]
        # x_one_hot = self.to_one_hot(x, x_mask)
        # x_prior = self.scale_shift(x_one_hot)
        #
        # # [batch_size, seq_len] -> [batch_size, seq_len, embedding_size]
        # x = self.embedding(x)
        # y = self.embedding(y)
        #
        # # let torch do sort, so we set enforce_sorted = False
        # x = nn.utils.rnn.pack_padded_sequence(x, lengths = len_x, batch_first = True, enforce_sorted = False)
        # x, (h_x, c_x) = self.lstm_x(x)
        # # x shape [batch_size, seq_len_x, hidden_size]
        # x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        # x = self.layer_norm_x(x)
        #
        # y = nn.utils.rnn.pack_padded_sequence(y, lengths = len_y, batch_first = True, enforce_sorted = False)
        # y, (h_y, c_y) = self.lstm_y(y)
        # # y shape [batch_size, seq_len_y, hidden_size]
        # y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first = True)
        # y = self.layer_norm_y(y)
        # print(decoder_outputs.shape)
        # print(top_vec.shape)
        # xy shape [batch_size, seq_len_y, hidden_size]
        # xy = self.attention(qx = decoder_outputs,
        #                     kx = top_vec,
        #                     vx = top_vec,
        #                     v_mask = mask_src)
        #
        # # xy shape [batch_size, seq_len_y, 2*hidden_size]
        # xy = torch.cat((decoder_outputs, xy), dim = -1)
        # # xy shape [batch_size, seq_len_y, embedding_size]
        # xy = self.linear_1(xy)
        # xy = self.activation(xy)
        # # xy shape [batch_size, seq_len_y, vocab_size + 4]
        # xy = self.linear_2(xy)
        # # xy shape [batch_size, seq_len_y, vocab_size + 4]
        # xy = (xy + top_vec) / 2
        # xy = self.softmax(xy)
        # # xy shape [batch_size, seq_len_y, vocab_size + 4]
        # return xy

        return decoder_outputs, None, [copy_prob[0], copy_prob[1], copy_prob[2], z_top_vec[:, 0, :]]

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

