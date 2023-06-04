import json
from data import load_data, get_epoch
import model
import torch
from torch import nn
import os
from random import shuffle
import argparse
import optuna
import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)


device = 'cuda:2'


def train_epoch(model, data, config, optimizer):
    model.train()
    n_iter = 0
    epoch_x, epoch_y, lengths_x = get_epoch(data["train_x"], data["train_y"], config["batch_size"], is_train=True)
    epoch_loss = 0
    corrects = 0
    criterion = nn.CrossEntropyLoss()
    # corrects_neg, corrects_pos = 0, 0

    for batch_x, batch_y, length_x in zip(epoch_x, epoch_y, lengths_x):
        batch_x = torch.LongTensor(batch_x)
        batch_y = torch.LongTensor(batch_y)
        lengths_x = torch.LongTensor(length_x)

        if config["cuda"]:
            batch_x, batch_y, lengths_x = batch_x.cuda(device), batch_y.cuda(device), lengths_x.cuda(device)

        optimizer.zero_grad()
        pred = model(batch_x)['logits']
        loss = criterion(pred, batch_y)
        n_iter += 1
        epoch_loss += float(loss)
        loss.backward()
        optimizer.step()

        batch_corrects = int((torch.max(pred, 1)[1].view(batch_y.size()).data == batch_y.data).sum())
        corrects += batch_corrects

        # if n_iter % 200 == 0:
        #     eval()
        #     model.train()

    return epoch_loss / len(data["train_y"]), corrects / len(data["train_y"]) * 100


def eval_epoch(model, data, config):
    model.eval()
    n_iter = 0
    epoch_x, epoch_y, lengths_x = get_epoch(data["valid_x"], data["valid_y"], config["batch_size"], is_train=False)
    epoch_loss = 0
    corrects = 0
    criterion = nn.CrossEntropyLoss()

    for batch_x, batch_y, length_x in zip(epoch_x, epoch_y, lengths_x):
        batch_x = torch.LongTensor(batch_x)
        batch_y = torch.LongTensor(batch_y)
        lengths_x = torch.LongTensor(length_x)

        if config["cuda"]:
            batch_x, batch_y, lengths_x = batch_x.cuda(device), batch_y.cuda(device), lengths_x.cuda(device)

        # optimizer.zero_grad()
        pred = model(batch_x)['logits']
        loss = criterion(pred, batch_y)
        n_iter += 1
        epoch_loss += float(loss)

        batch_corrects = int((torch.max(pred, 1)[1].view(batch_y.size()).data == batch_y.data).sum())
        corrects += batch_corrects

        # if n_iter % 200 == 0:
        #     eval()
        #     model.train()
        del batch_x, batch_y, pred, loss

    return epoch_loss / len(data["valid_y"]), corrects / len(data["valid_y"]) * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as fp:
        config = json.load(fp)

    data = load_data(config=config)
    with open(config["model_path"] + "/w2i.json", "w") as fp:
        json.dump(data["word_to_idx"], fp)

    study = optuna.create_study(direction='maximize')

    all_metrics = {}

    def objective(trial):
        embedding_dim = trial.suggest_int('embedding_dim', 512, 512)
        num_filters = trial.suggest_int('num_filters', 32, 32)
        learning_rate = trial.suggest_float('learning_rate', 2e-4, 2e-4, log=True)
        model_instance = model.CnnClassifier(ngram_sizes=config["ngram_sizes"], embedding_dim=embedding_dim,
                                             num_filters=num_filters, padding_idx=data["word_to_idx"]["@@PAD@@"],
                                             num_classes=max(data["classes"])+1, vocab_size=len(data["vocab"]))

        if "cuda" not in config:
            config["cuda"] = False

        if config["cuda"]:
            model_instance = model_instance.cuda(device)

        parameters = filter(lambda p: p.requires_grad, model_instance.parameters())
        optimizer = torch.optim.Adam(parameters, learning_rate)

        if not os.path.exists(config["model_path"]):
            os.makedirs(config["model_path"])

        print("\t".join(["Epoch", "Loss", "Acc", "Eval", "Acc", "Best"]))
        metrics = {"loss": [], "acc": [], "eval": [], "eval_acc": [], "best": -1}

        for I in range(config["num_epochs"]):
            loss, acc = train_epoch(model=model_instance, data=data, config=config, optimizer=optimizer)
            eval_loss, eval_acc = eval_epoch(model=model_instance, data=data, config=config)

            metrics["loss"].append(loss)
            metrics["acc"].append(acc)
            metrics["eval"].append(eval_loss)
            metrics["eval_acc"].append(eval_acc)

            if eval_acc > metrics["best"]:
                metrics["best"] = eval_acc
            if len(study.trials) == 1 or eval_acc > study.best_value:
                torch.save(model_instance, config["model_path"] + "/model")

            print(f"{I}\t{loss:.5f}\t{acc:.2f}\t{eval_loss:.5f}\t{eval_acc:.2f}\t{metrics['best']:.2f}")
        
        all_metrics[str(trial.params)] = metrics
        return metrics['best']

    study.optimize(objective, n_trials=1)
    with open(config["model_path"] + "/metrics.json", "w") as fp:
        json.dump(all_metrics, fp)

