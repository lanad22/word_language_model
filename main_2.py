# main_2.py

import time
import math
import torch
import torch.nn as nn
import data
import model
import os

class Config:
    def __init__(self):
        self.data = './data/wikitext-2'
        self.model = 'LSTM'  # Options: 'RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU', 'Transformer'
        self.emsize = 200
        self.nhid = 200
        self.nlayers = 2
        self.lr = 20.0
        self.clip = 0.25
        self.epochs = 40
        self.batch_size = 20
        self.bptt = 35
        self.dropout = 0.2
        self.tied = False
        self.seed = 1111
        self.cuda = False
        self.mps = False
        self.log_interval = 200
        self.save = 'model.pt'
        self.onnx_export = ''
        self.nhead = 2
        self.dry_run = False

def batchify(data, bsz, device):
    # Your existing batchify function
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, bptt):
    # Your existing get_batch function
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):
    # Your existing repackage_hidden function
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(data_source, mdl, criterion, args, corpus):
    # Your existing evaluate function
    mdl.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    eval_batch_size = 10
    if args.model != 'Transformer':
        hidden = mdl.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
            if args.model == 'Transformer':
                output = mdl(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = mdl(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

def train(args,mdl,train_data,corpus,criterion,lr,epoch):
    # Turn on training mode which enables dropout.
    mdl.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = mdl.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        mdl.zero_grad()
        if args.model == 'Transformer':
            output = mdl(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = mdl(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), args.clip)
        for p in mdl.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

def export_onnx(mdl, args, device, path, batch_size):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    mdl.eval()
    seq_len = args.bptt
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = mdl.init_hidden(batch_size)
    torch.onnx.export(mdl, (dummy_input, hidden), path)

def main(args):
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    corpus = data.Corpus(args.data)

    eval_batch_size = 10

    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    ntokens = len(corpus.dictionary)

    # Correctly instantiate the model with the new variable name 'mdl'
    if args.model == 'Transformer':
        mdl = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
    else:
        mdl = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

    criterion = nn.NLLLoss()
    # Loop over epochs.
    lr = args.lr
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(args,mdl,train_data,corpus,criterion,lr,epoch)
            val_loss = evaluate(val_data, mdl, criterion, args, corpus)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(mdl, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        mdl = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            mdl.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data, mdl, criterion, args, corpus)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(mdl, args, device, args.onnx_export, batch_size=1)
    return math.exp(test_loss)

if __name__ == '__main__':
    config = Config()
    main(config)
