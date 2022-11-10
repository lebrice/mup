"""
PyTorch Wikitext-2 Transformer Language Model, with μP.

To train a μP model, one needs to first specify the base shapes. To save base shapes info, run, for example,

    python main.py --d_model 256 --save_base_shapes width256.bsh

To train using MuAdam, run

    python main.py --d_model 256 --load_base_shapes width256.bsh --cuda --optimizer muadam

To perform coord check, run

    python main.py --load_base_shapes width256.bsh --optimizer sgd --lr 0.5 --cuda --coord_check

    python main.py --load_base_shapes width256.bsh --optimizer adam --lr 0.01 --cuda --coord_check

If you don't specify a base shape file, then you are using standard parametrization

    python main.py --d_model 256 --cuda --optimizer muadam

Note that models of different depths need separate `.bsh` files.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from apex import amp
except:
    print("Failed to import apex. You can still train with --precision {float|double}.")

from mup.coord_check import get_coord_data, plot_coord_data
from mup import MuAdam, MuSGD, get_shapes, make_base_shapes, set_base_shapes

import data
import model as mdl
import wandb

###############################################################################
# Training code
###############################################################################

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def batchloader(train_data, bptt):
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        yield get_batch(train_data, i, bptt)


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def setprec(t, precision):
    if precision == "half":
        # do nothing since this is handled by AMP
        return t
    elif precision == "float":
        return t.float()
    elif precision == "double":
        return t.double()
    else:
        raise ValueError(f"invalid precision string {args.precision}")


def coord_check(
    mup, lr, optimizer, batch_size, nsteps, nseeds, data_dir, args, plotdir="", legend=False
):

    corpus = data.Corpus(data_dir)
    ntokens = len(corpus.dictionary)

    def gen(w, standparam=False):
        import model as _model

        def f():
            model = _model.TransformerModel(
                args,
                ntokens,
                ninp=w,
                nhead=args.nhead,
                nhid=w * args.ffn_ratio,
                nlayers=args.nlayers,
                dropout=args.dropout,
                tied=args.tied,
                bias=args.bias,
                encoder_var=args.init_var,
                decoder_var=args.init_var,
                standparam=standparam,
            ).to(args.device)
            model = setprec(model, args.precision)
            if standparam:
                set_base_shapes(model, None)
            else:
                assert args.load_base_shapes, "load_base_shapes needs to be nonempty"
                set_base_shapes(model, args.load_base_shapes)
            return model

        return f

    optimizer = optimizer.replace("mu", "")
    widths = 2 ** np.arange(7, 14 if optimizer == "sgd" else 12)
    models = {w: gen(w, standparam=not mup) for w in widths}

    train_data = batchify(corpus.train, batch_size, device=args.device)
    df = get_coord_data(
        models,
        batchloader(train_data, args.bptt),
        mup=mup,
        lr=lr,
        optimizer=optimizer,
        flatten_output=True,
        nseeds=nseeds,
        nsteps=nsteps,
        lossfn="nll",
    )

    prm = "μP" if mup else "SP"
    return plot_coord_data(
        df,
        legend=legend,
        save_to=os.path.join(plotdir, f"{prm.lower()}_trsfmr_{optimizer}_coord.png"),
        suptitle=f"{prm} Transformer {optimizer} lr={lr} nseeds={nseeds}",
        face_color="xkcd:light grey" if not mup else None,
    )


from dataclasses import dataclass
from simple_parsing.helpers import field
from typing_extensions import Literal


@dataclass
class Args:

    data: str = "./data/wikitext-2"
    """ location of the data corpus """

    bias: bool = False
    """ use bias """

    save_base_shapes: str = ""
    """file location to save base shapes at """

    load_base_shapes: str = ""
    """file location to load base shapes from """

    d_model: int = 256
    """width of the model"""

    ffn_ratio: int = 1
    """the ratio of d_ffn to d_model"""

    nlayers: int = 2
    """number of layers"""

    nhead: int = 2
    """the number of heads in the encoder/decoder of the transformer model"""

    lr: float = 0.001
    """initial learning rate"""
    momentum: float = 0
    """momentum"""
    output_mult: float = 1
    """output is multiplied by sqrt(output_mult/d_model)"""

    input_mult: float = 1
    """input is multiplied by sqrt(input_mult*d_model)"""

    attn_mult: float = 1
    """attn is multiplied by sqrt(attn_mult)/head_dim"""

    optimizer: Literal["sgd", "musgd", "adam", "muadam"] = "musgd"

    init_var: float = 1
    """weights are initialized with variance init_var/ninp"""

    clip: float = 0.25
    """gradient clipping"""
    epochs: int = 40
    """upper epoch limit"""
    batch_size: int = field(default=20, metavar="N", help="batch size")

    bptt: int = 35
    """sequence length """

    dropout: float = 0.2
    """dropout applied to layers (0 = no dropout)"""

    tied: bool = False
    """tie the word embedding and softmax weights"""

    seed: int = 1111
    """random seed"""

    cuda: bool = False
    """use CUDA"""

    precision: Literal["float", "double", "half"] = "float"
    """ Precision to use: float | double | half"""

    log_interval: int = field(default=200, metavar="N")
    """report interval"""

    save_dir: str | None = None
    """path to save the final model"""

    resume_dir: str | None = None
    """path to resume training"""

    log_dir: str = "."
    """path to save logs"""

    coord_check: bool = False
    """test μ parametrization is correctly implemented by collecting statistics on coordinate
    distributions for a few steps of training."""

    coord_check_nsteps: int = 3
    """Do coord check with this many steps."""

    coord_check_nseeds: int = 3
    """number of seeds for testing correctness of μ parametrization"""

    def __post_init__(self):
        self.device: torch.device = torch.device("cuda" if self.cuda else "cpu")


import simple_parsing

import dataclasses
from simple_parsing import ArgumentParser
from dataclasses import dataclass


def main():

    parser = simple_parsing.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_arguments(Args, dest="args")
    namespace = parser.parse_args()

    args: Args = namespace.args

    wandb.init(project="mup_original", config=dataclasses.asdict(args))

    print(args)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = args.device = torch.device("cuda" if args.cuda else "cpu")

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data)

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(corpus.dictionary)

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.0
        ntokens = len(corpus.dictionary)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i, args.bptt)
                output = model(data)
                output = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)

    def train(optimizer, epoch):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.0
        epoch_loss = 0.0
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        first_loss = None
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            data, targets = get_batch(train_data, i, args.bptt)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.

            optimizer.zero_grad()
            output = model(data)
            output = output.view(-1, ntokens)
            loss = criterion(output, targets)
            if torch.isnan(loss):
                exit(0)
            if args.precision == "half":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.clip > 0:
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                if args.precision == "half":
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            total_loss += loss.item()
            epoch_loss += len(data) * loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | "
                    "loss {:5.2f} | ppl {:8.2f}".format(
                        epoch,
                        batch,
                        len(train_data) // args.bptt,
                        lr,
                        elapsed * 1000 / args.log_interval,
                        cur_loss,
                        np.exp(cur_loss),
                    )
                )
                wandb.log(
                    dict(
                        epoch=epoch,
                        learning_rate=lr,
                        loss=loss.item(),
                        cur_loss=cur_loss,
                        perplexity=np.exp(cur_loss),
                    )
                )
                total_loss = 0
                start_time = time.time()
                if first_loss is None:
                    first_loss = cur_loss

        return epoch_loss / (len(train_data) - 1), first_loss

    if args.coord_check:
        print("testing parametrization")

        os.makedirs("coord_checks", exist_ok=True)
        plotdir = "coord_checks"
        coord_check(
            mup=True,
            lr=args.lr,
            optimizer=args.optimizer,
            batch_size=args.batch_size,
            nsteps=args.coord_check_nsteps,
            nseeds=args.coord_check_nseeds,
            data_dir=args.data,
            args=args,
            plotdir=plotdir,
            legend=False,
        )
        coord_check(
            mup=False,
            lr=args.lr,
            optimizer=args.optimizer,
            batch_size=args.batch_size,
            nsteps=args.coord_check_nsteps,
            nseeds=args.coord_check_nseeds,
            data_dir=args.data,
            args=args,
            plotdir=plotdir,
            legend=False,
        )
        import sys

        sys.exit()

    model = mdl.TransformerModel(
        args,
        ntokens,
        ninp=args.d_model,
        nhead=args.nhead,
        nhid=args.d_model * args.ffn_ratio,
        nlayers=args.nlayers,
        dropout=args.dropout,
        tied=args.tied,
        bias=args.bias,
        encoder_var=args.init_var,
        decoder_var=args.init_var,
        standparam=args.load_base_shapes == "",
    )
    if args.save_base_shapes:
        print(f"saving base shapes at {args.save_base_shapes}")
        base_shapes = get_shapes(model)
        delta_shapes = get_shapes(
            # just need to change whatever dimension(s) we are scaling
            mdl.TransformerModel(
                args,
                ntokens,
                ninp=args.d_model * 2,
                nhead=args.nhead,
                nhid=args.d_model * args.ffn_ratio * 2,
                nlayers=args.nlayers,
                dropout=args.dropout,
                tied=args.tied,
                bias=args.bias,
                encoder_var=args.init_var,
                decoder_var=args.init_var,
                standparam=args.load_base_shapes == "",
            )
        )
        make_base_shapes(base_shapes, delta_shapes, savefile=args.save_base_shapes)
        print("done and exit")
        import sys

        sys.exit()
    if args.load_base_shapes:
        print(f"loading base shapes from {args.load_base_shapes}")
        set_base_shapes(model, args.load_base_shapes)
        print("done")
    else:
        print(f"using own shapes")
        set_base_shapes(model, None)
        print("done")

    model = model.to(device)
    model = setprec(model, args.precision)

    criterion = nn.NLLLoss()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    # Loop over epochs.
    lr = args.lr
    best_val_loss = float("inf")

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "musgd":
        optimizer = MuSGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "muadam":
        optimizer = MuAdam(model.parameters(), lr=args.lr)
    else:
        raise ValueError()

    # half-precision black magic
    if args.precision == "half":
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O1", min_loss_scale=0.0001, verbosity=0
        )

    logs = []
    start_epoch = 0
    if args.resume_dir and os.path.exists(os.path.join(args.resume_dir, "checkpoint_last.pt")):
        checkpoint = torch.load(os.path.join(args.resume_dir, "checkpoint_last.pt"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.precision == "half":
            amp.load_state_dict(checkpoint["amp"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        logs = checkpoint["logs"]

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss, first_loss = train(optimizer, epoch)
            # print(first_loss)
            val_loss = evaluate(val_data)
            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f}".format(
                    epoch, (time.time() - epoch_start_time), val_loss, np.exp(val_loss)
                )
            )
            print("-" * 89)
            logs.append(
                dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, first_loss=first_loss)
            )
            # Save the model if the validation loss is the best we've seen so far.
            if args.save_dir is not None:
                if val_loss < best_val_loss:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "logs": logs,
                    }
                    if args.precision == "half":
                        checkpoint["amp"] = (amp.state_dict(),)
                    with open(os.path.join(args.save_dir, "checkpoint_best.pt"), "wb") as f:
                        torch.save(checkpoint, f)
                    best_val_loss = val_loss
                else:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "logs": logs,
                    }
                    if args.precision == "half":
                        checkpoint["amp"] = amp.state_dict()
                with open(os.path.join(args.save_dir, "checkpoint_last.pt"), "wb") as f:
                    torch.save(checkpoint, f)

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    # Load the best saved model.
    if args.save_dir is not None:
        with open(os.path.join(args.save_dir, "checkpoint_best.pt"), "rb") as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if args.precision == "half":
                amp.load_state_dict(checkpoint["amp"][0])
        # Run on test data.
        test_loss = evaluate(test_data)
        print("=" * 89)
        print(
            "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
                test_loss, np.exp(test_loss)
            )
        )
        print("=" * 89)
        wandb.log({"test_loss": test_loss, "test_ppl": np.exp(test_loss)})
        logs.append(dict(epoch="-1", test_loss=test_loss))

    with open(os.path.join(os.path.expanduser(args.log_dir), "logs.tsv"), "w") as f:
        logdf = pd.DataFrame(logs)
        print(os.path.join(os.path.expanduser(args.log_dir), "logs.tsv"))
        f.write(logdf.to_csv(sep="\t", float_format="%.4f"))

    from orion.client import report_objective

    report_objective(test_loss)


if __name__ == "__main__":
    main()
