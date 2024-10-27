import torch
import time
import torch.nn.functional as F
import logging
def train(ViewW , AW, model, criterion,criterion1, optimizer, epoch, args):
    pos_dist = 0  # mean distance of pos. pairs
    neg_dist = 0
    false_neg_dist =  0  # mean distance of false neg. pairs (pairs in noisy labels)
    true_neg_dist = 0
    pos_count = 0  # count of pos. pairs
    neg_count = 0
    false_neg_count = 0  # count of neg. pairs (pairs in noisy labels)
    true_neg_count = 0

    if epoch % 10 == 0:
        logging.info("=======> Train epoch: {}/{}".format(epoch, args.epochs))
    model.train()
    time0 = time.time()
    loss_value = 0

        # labels refer to noisy labels for the constructed pairs, while real_labels are the clean labels for these pairs
    ViewW , AW = ViewW.to(args.gpu), AW.to(args.gpu)
    # try:
    h0, h1 = model(AW[0],ViewW[0]),model(AW[1],ViewW[1])
    # except:
    #     print("error raise in batch", batch_idx)
    #
    loss = criterion(h0, h1)

    loss_value += loss.item()
    if epoch != 0:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_time = time.time() - time0


    return pos_dist, neg_dist, false_neg_dist, true_neg_dist, epoch_time