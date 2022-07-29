import torch
import pandas as pd
from models.model_lstm import LSTM_Model
from loss import SimpleCCELoss
from utils import load_vocab, evaluate
import config
import time
import glob
import os
import argparse
import random
import signal
from models.predictor import build_predictor
from models.model_builder import Z_AbsSummarizer
from others.logging import logger, init_logger
import random
from models import data_loader, model_lstm
from models.data_loader import load_dataset
from pytorch_transformers import BertTokenizer
from models.trainer import build_trainer
COPY=False
if COPY:
    from models.loss_copy import abs_loss
else:
    from models.loss import abs_loss
model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

def train_abs_single(args, device_id):
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    if (args.load_from_extractive != ''):
        logger.info('Loading bert from extractive model %s' % args.load_from_extractive)
        bert_from_extractive = torch.load(args.load_from_extractive, map_location=lambda storage, loc: storage)
        bert_from_extractive = bert_from_extractive['model']
    else:
        bert_from_extractive = None
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True, is_test=False)

    model = Z_AbsSummarizer(args, device, checkpoint, bert_from_extractive)
    if (args.sep_optim):
        optim_bert = model_lstm.build_optim_bert(args, model, checkpoint)
        optim_dec = model_lstm.build_optim_dec(args, model, checkpoint)
        optim = [optim_bert, optim_dec]
    else:
        optim = [model_lstm.build_optim(args, model, checkpoint)]

    logger.info(model)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True, cache_dir=args.temp_dir)
    # symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
    #            'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
               'PAD': tokenizer.vocab['[PAD]']}

    if COPY:
        train_loss = abs_loss(model.generator, symbols, model.vocab_size, device, train=True,
                              label_smoothing=args.label_smoothing, copy_generator=model.copy_generator)
    else:
        train_loss = abs_loss(model.generator, symbols, model.vocab_size, device, train=True,
                              label_smoothing=args.label_smoothing)

    trainer = build_trainer(args, device_id, model, optim, train_loss)
    trainer.train(train_iter_fct, args.train_steps)


def validate_abs(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            if (args.test_start_from != -1 and step < args.test_start_from):
                xent_lst.append((1e6, cp))
                continue
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:5]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test_abs(args, device_id, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test_abs(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)

def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = Z_AbsSummarizer(args, device, checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True, cache_dir=args.temp_dir)
    # symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
    #            'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused3]']}
    symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
               'PAD': tokenizer.vocab['[PAD]']}

    if COPY:
        valid_loss = abs_loss(model.generator, symbols, model.vocab_size, train=False, device=device, copy_generator=model.copy_generator)
    else:
        valid_loss = abs_loss(model.generator, symbols, model.vocab_size, train=False, device=device)

    trainer = build_trainer(args, device_id, model, None, valid_loss)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()




def test_abs(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = Z_AbsSummarizer(args, device, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True, cache_dir=args.temp_dir)
    # symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
    #            'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused3]']}
    symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
               'PAD': tokenizer.vocab['[PAD]']}
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step)


def test_text_abs(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = Z_AbsSummarizer(args, device, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True, cache_dir=args.temp_dir)
    # symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
    #            'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused3]']}
    symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
               'PAD': tokenizer.vocab['[PAD]']}
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step)


def baseline(args, cal_lead=False, cal_oracle=False):
    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.batch_size, 'cpu',
                                       shuffle=False, is_test=True)

    trainer = build_trainer(args, '-1', None, None, None)
    #
    if (cal_lead):
        trainer.test(test_iter, 0, cal_lead=True)
    elif (cal_oracle):
        trainer.test(test_iter, 0, cal_oracle=True)



# def main():
#     data = pd.read_csv(config.train_dataset_file)
#     train_data = data.iloc[20:]
#     validation_data = data.iloc[:20]
#
#     chars, id2char, char2id = load_vocab(config.vocabulary_file, config.train_dataset_file)
#
#     vocab_size = len(chars)  # 5830, Does NOT include 4 special chars
#
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     print("Device check. You are using:", device)
#
#     train_dataset = MyDataset(data = train_data,
#                               char2id = char2id)
#     train_data_loader = torch.utils.data.DataLoader(train_dataset,
#                                    batch_size = config.batch_size,
#                                    shuffle = True,
#                                    collate_fn = my_collate_fn)
#
#     validation_dataset = MyDataset(data = validation_data,
#                                    char2id = char2id,
#                                    convert_to_ids = False)
#     validation_data_loader = torch.utils.data.DataLoader(validation_dataset,
#                                    batch_size = 1,
#                                    shuffle = False)
#
#     model = LSTM_Model(vocab_size = vocab_size,
#                        embedding_size = config.embedding_size,
#                        hidden_size = config.hidden_size,
#                        device = device).to(device)
#
#     criterion = SimpleCCELoss(device = device)
#     optimizer = torch.optim.Adam(params = model.parameters())
#     epochs = config.epochs
#
#     best_training_loss = float('inf')
#     for epoch in range(epochs):
#         # training part
#         running_loss = 0.0
#         epoch_loss = 0.0
#         for i, data in enumerate(train_data_loader):
#             x, y, len_x, len_y = data
#             x, y, len_x, len_y = x.to(device), y.to(device), len_x, len_y
#
#             model.train()
#             output = model(x, y, len_x, len_y)
#             loss = criterion(output, y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#             epoch_loss += loss.item()
#             if i!=0 and i%50 == 0:
#                 print('Epoch {}/{}, Step {} - loss: {}'.format(epoch+1, epochs, i, running_loss/50))
#                 running_loss = 0.0
#
#         torch.save(model.state_dict(), 'saved_weight/saved_model_on_epoch_{}.pth'.format(epoch+1))
#         epoch_loss /= len(train_data_loader)
#         if epoch_loss < best_training_loss:
#             best_training_loss = epoch_loss
#             torch.save(model.state_dict(), 'saved_weight/best_model_training.pth')
#         print('Training loss: {}, Best training loss: {}'.format(epoch_loss, best_training_loss))
#
#
#         # validation part
#         evaluate(validation_data_loader, model, device, char2id, id2char)
#
#
# if __name__ == '__main__':
#     main()