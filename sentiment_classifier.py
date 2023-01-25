import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from scripts.args import parse_args
import time
from data.dataset import TweetsDataset
from scripts.net import TextClassificationModel

def main():

    args = parse_args()

    # Hyperparameters
    EPOCHS = args.epochs # epoch
    LR = args.learning  # learning rate
    BATCH_SIZE = 10 # batch size for training

    test_size = args.test
    train_size = 27480 - test_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    tweets = TweetsDataset(train_size)
    tweets.to_device(device)

    num_class = len(set([label for (label, _) in tweets.train_iter]))
    vocab_size = len(tweets.vocab)
    emsize = 10

    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None

    train_dataloader, valid_dataloader = tweets.make_dataloaders(BATCH_SIZE)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        model.fit(train_dataloader, epoch, optimizer)
        accu_val = model.evaluate(valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch,
                                            time.time() - epoch_start_time,
                                            accu_val))
        print('-' * 59)

if __name__ == '__main__':
    main()