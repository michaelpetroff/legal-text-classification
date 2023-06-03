import numpy as np
import os
import json
import requests
import re
from tqdm.auto import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, RobertaTokenizer, XLMRobertaTokenizer
from torch.utils.tensorboard import SummaryWriter


train_data = pd.read_csv('train_trans_DMO_2.csv')
texts_train, targets_train = train_data.text, train_data.target
val_data = pd.read_csv('val_trans_DMO_2.csv')
texts_val, targets_val = val_data.text, val_data.target


class LabelSmoothingLoss(nn.Module):
     
    def __init__(self,
                 smoothing: float = 0.1,
                 use_kl: bool = False,
                 ignore_index: int = -100):
        super().__init__()

        assert 0 <= smoothing < 1

        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.use_kl = use_kl

    def smooth_one_hot(self, true_labels: torch.Tensor, classes: int) -> torch.Tensor:

        confidence = 1.0 - self.smoothing

        with torch.no_grad():
            true_dist = torch.empty(size=(true_labels.size(0), classes), device=true_labels.device)
            true_dist.fill_(self.smoothing / (classes - 1))
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)

        return true_dist

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        :param prediction: [batch_size, num_classes]
        :param target: [batch_size]
        :return: scalar
        """

        prediction = F.log_softmax(prediction, dim=-1)

        target_smoothed_dist = self.smooth_one_hot(target, classes=prediction.size(-1))

        if self.use_kl:
            loss = F.kl_div(prediction, target_smoothed_dist, reduction='batchmean')
        else:
            loss = torch.mean(torch.sum(-target_smoothed_dist * prediction, dim=-1))

        return loss


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, targets):
        self.texts, self.targets = texts, targets
        assert len(texts) == len(targets)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx]


def add_new_token(tokenizer, backbone_model_name, token):

    # update the vocabulary with the new token and the 'Ġ'' version
    roberta_vocab = tokenizer.get_vocab()
    roberta_vocab.update({token : len(roberta_vocab)}) 
    roberta_vocab.update({chr(288) + token : len(roberta_vocab)}) # chr(288) = 'Ġ'
    with open('vocab.tmp', 'w', encoding = 'utf-8') as tmp_vocab_file:
        json.dump(roberta_vocab, tmp_vocab_file, ensure_ascii=False)

    # get and modify the merges file so that the new token will always be tokenized as a single word
    url = f'https://huggingface.co/{backbone_model_name}/resolve/main/merges.txt'
    roberta_merges = requests.get(url).content.decode().split('\n')

    # this is a helper function to loop through a list of new tokens and get the byte-pair encodings
    # such that the new token will be treated as a single unit always
    def get_roberta_merges_for_new_tokens(new_tokens):
        merges = [gen_roberta_pairs(new_token) for new_token in new_tokens]
        merges = [pair for token in merges for pair in token]
        return merges

    def gen_roberta_pairs(new_token, highest = True):
        # highest is used to determine whether we are dealing with the Ġ version or not. 
        # we add those pairs at the end, which is only if highest = True
        
        # this is the hard part...
        chrs = [c for c in new_token] # list of characters in the new token, which we will recursively iterate through to find the BPEs
        
        # the simplest case: add one pair
        if len(chrs) == 2:
            if not highest: 
                return tuple([chrs[0], chrs[1]])
            else:
                return [' '.join([chrs[0], chrs[1]])]
        
        # add the tokenization of the first letter plus the other two letters as an already merged pair
        if len(chrs) == 3:
            if not highest:
                return tuple([chrs[0], ''.join(chrs[1:])])
            else:
                return gen_roberta_pairs(chrs[1:]) + [' '.join([chrs[0], ''.join(chrs[1:])])]
        
        if len(chrs) % 2 == 0:
            pairs = gen_roberta_pairs(''.join(chrs[:-2]), highest = False)
            pairs += gen_roberta_pairs(''.join(chrs[-2:]), highest = False)
            pairs += tuple([''.join(chrs[:-2]), ''.join(chrs[-2:])])
            if not highest:
                return pairs
        else:
            # for new tokens with odd numbers of characters, we need to add the final two tokens before the
            # third-to-last token
            pairs = gen_roberta_pairs(''.join(chrs[:-3]), highest = False)
            pairs += gen_roberta_pairs(''.join(chrs[-2:]), highest = False)
            pairs += gen_roberta_pairs(''.join(chrs[-3:]), highest = False)
            pairs += tuple([''.join(chrs[:-3]), ''.join(chrs[-3:])])
            if not highest:
                return pairs
        
        pairs = tuple(zip(pairs[::2], pairs[1::2]))
        pairs = [' '.join(pair) for pair in pairs]
        
        # pairs with the preceding special token
        g_pairs = []
        for pair in pairs:
            if re.search(r'^' + ''.join(pair.split(' ')), new_token):
                g_pairs.append(chr(288) + pair)
        
        pairs = g_pairs + pairs
        pairs = [chr(288) + ' ' + new_token[0]] + pairs
        
        pairs = list(dict.fromkeys(pairs)) # remove any duplicates
        
        return pairs

    # first line of this file is a comment; add the new pairs after it
    roberta_merges = roberta_merges[:1] + get_roberta_merges_for_new_tokens([token]) + roberta_merges[1:]
    roberta_merges = list(dict.fromkeys(roberta_merges))
    with open('merges.tmp', 'w', encoding = 'utf-8') as tmp_merges_file:
        tmp_merges_file.write('\n'.join(roberta_merges))

    return RobertaTokenizer(name_or_path=backbone_model_name, vocab_file='vocab.tmp', merges_file='merges.tmp')


class Collater:
    def __init__(self, model_name, max_length=64):
        # self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        for token in ['ORG', 'MONEY_0', 'MONEY_1', 'MONEY_2', 'MONEY_F']:
            self.tokenizer = add_new_token(self.tokenizer, model_name, token)
        self.max_length = max_length
        
    def __call__(self, batch):
        texts, targets = list(), list()
        
        for text, target in batch:
            texts.append(text)
            targets.append(target)
            
        batch = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        token_type_mask = batch['input_ids']
        sep_indices = token_type_mask.max(dim=1).indices + 1
        token_type_ids = torch.arange(batch['input_ids'].size(1)).unsqueeze(0).repeat(batch['input_ids'].size(0), 1)
        token_type_ids = (token_type_ids > sep_indices.unsqueeze(1)).long()
        batch['token_type_ids'] = token_type_ids

        return batch, targets

def train(model, loader, criterion, optimizer, device,
          losses, predictions, targets,
          grad_clip, validate_freq, accumulate_freq, use_tqdm):
    assert len(targets) == len(predictions) and len(losses) <= len(loader)
    model.train()

    if use_tqdm:
        progress_bar = tqdm(total=len(loader), desc='Training')

    for i, (batch, cur_targets) in enumerate(loader):
        for key, value in batch.items():
            batch[key] = value.to(device)
        cur_targets = torch.tensor(cur_targets, dtype=torch.long, device=device)
        logits = model(**batch).logits

        if isinstance(criterion, nn.BCEWithLogitsLoss):
            loss = torch.tensor(0.0, requires_grad=True)
            contexts = np.array(contexts)
            for context in np.unique(contexts):
                mask = contexts == context
                if True in mask and False in mask:
                    output = torch.softmax(logits[mask, 1], dim=0)
                else:
                    output = logits[mask, 1]
                loss = loss + criterion(output, cur_targets[mask].float())
            loss = loss / len(batch)
        elif isinstance(criterion, (nn.CrossEntropyLoss, LabelSmoothingLoss)):
            loss = criterion(logits, cur_targets)
        else:
            raise ValueError('Bad criterion!')

        if (i+1) % accumulate_freq == 0 or i+1 == len(loader):
            optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip)
        if (i+1) % accumulate_freq == 0 or i+1 == len(loader):
            optimizer.step()

        cur_predictions = torch.argmax(logits, -1).detach().cpu().numpy()
        cur_targets = cur_targets.cpu().numpy()
        cur_loss = loss.item()
        if len(losses) <= i:
            predictions = np.concatenate([predictions, cur_predictions])
            targets = np.concatenate([targets, cur_targets])
            losses = np.append(losses, cur_loss)
        else:
            predictions[i*loader.batch_size:i*loader.batch_size+len(cur_predictions)] = cur_predictions
            targets[i*loader.batch_size:i*loader.batch_size+len(cur_targets)] = cur_targets
            losses[i] = cur_loss

        if use_tqdm:
            progress_bar.update()
            progress_bar.set_postfix(loss=np.mean(losses[max(0, i-100):i]) if i>0 else np.nan)

        if validate_freq > 0 and (i+1) % validate_freq == 0 and len(loader) - (i+1) > validate_freq:
            yield losses, predictions, targets, i+1

    if use_tqdm:
        progress_bar.close()

    yield losses, predictions, targets, len(loader)


@torch.inference_mode()
def validate(model, loader, criterion, device, use_tqdm):
    model.eval()

    losses = list()
    predictions = list()

    if use_tqdm:
        progress_bar = tqdm(total=len(loader), desc='Validation')

    for batch, target in loader:

        for key, value in batch.items():
            batch[key] = value.to(device)
        target = torch.tensor(target, dtype=torch.long, device=device)
        logits = model(**batch).logits

        if isinstance(criterion, nn.BCEWithLogitsLoss):
            loss = criterion(logits, F.one_hot(target, num_classes=2).float())
        elif isinstance(criterion, (nn.CrossEntropyLoss, LabelSmoothingLoss)):
            loss = criterion(logits, target)
        else:
            raise ValueError('Bad criterion!')

        predictions.append(torch.argmax(logits, -1).detach().cpu().numpy())
        losses.append(loss.item())

        if use_tqdm:
            progress_bar.update()
            progress_bar.set_postfix(loss=np.mean(losses[-100:]))
    
    predictions = np.concatenate(predictions)

    if use_tqdm:
        progress_bar.close()

    return losses, predictions

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    backbone_model_name = 'xlm-roberta-base'
    max_length = 192
    batch_size = 32

    train_dataset = MyDataset(texts_train.to_list(), targets_train.to_list())
    train_collater = Collater(
        model_name=backbone_model_name,
        max_length=max_length
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, collate_fn=train_collater
    )

    valid_dataset = MyDataset(texts_val.to_list(), targets_val.to_list())
    valid_collater = Collater(model_name=backbone_model_name, max_length=max_length)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, collate_fn=valid_collater
    )


    model = AutoModelForSequenceClassification.from_pretrained(
        backbone_model_name, num_labels=38, 
        # hidden_dropout_prob=0.2
    )

    current_segment_embeddings = model.roberta.embeddings.token_type_embeddings.weight.data
    segment_embeddings = torch.cat(
        [
            current_segment_embeddings,
            current_segment_embeddings + torch.rand_like(current_segment_embeddings) * 0.01
        ]
    )
    model.roberta.embeddings.token_type_embeddings = model.roberta.embeddings.token_type_embeddings.from_pretrained(
        segment_embeddings,
        freeze=False
    )


    # NAME
    model_name = 'xlm_192_32_50_ls_adj_DMO'

    os.makedirs(f'finetuned_models/{model_name}', exist_ok=True)

    # SETTINGS
    gpu_id = 1
    device = torch.device(f'cuda:{gpu_id}') if gpu_id is not None else torch.device('cpu')
    model.to(device)

    epochs_num = 10
    grad_clip = 2.
    lr = 5e-5
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingLoss()

    validate_freq = 0.1
    if 0 < validate_freq < 1:
        validate_freq = int(validate_freq * len(train_loader))
    accumulate_freq = 1
    use_tqdm = False
    use_tensorboard = True
    if use_tensorboard:
        writer = SummaryWriter(f'runs/{model_name}')
    adjust_learning_rate = True
    if adjust_learning_rate:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=7)

    best_accuracy = 0.
    train_losses, train_predictions, train_targets = [], [], []
    valid_targets = np.array(valid_dataset.targets)
    for epoch in range(epochs_num):
        for train_losses, train_predictions, train_targets, step in train(
            model, train_loader, criterion, optimizer, device,
            train_losses, train_predictions, train_targets,
            grad_clip, validate_freq, accumulate_freq, use_tqdm
        ):
            valid_losses, valid_predictions = validate(model, valid_loader, criterion, device, use_tqdm)

            valid_accuracy = (valid_predictions == valid_targets).mean()
            train_accuracy = (train_predictions == train_targets).mean()

            if use_tensorboard:
                global_step = step + epoch * len(train_loader)
                writer.add_scalar('Loss/train', np.mean(train_losses), global_step)
                writer.add_scalar('Loss/valid', np.mean(valid_losses), global_step)
                writer.add_scalar('Accuracy/train', train_accuracy, global_step)
                writer.add_scalar('Accuracy/valid', valid_accuracy, global_step)

            message = [
                f'Step {step} at epoch {epoch}' if step < len(train_loader) else f'After epoch: {epoch}',
                'Train:',
                f'Loss: {np.mean(train_losses):.3f} | Accuracy: {train_accuracy:.3f}',
                'Valid:',
                f'Loss: {np.mean(valid_losses):.3f} | Accuracy: {valid_accuracy:.3f}'
            ]

            message = '\n'.join(message)
            print(message, flush=True)

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                torch.save(model.state_dict(), f'finetuned_models/{model_name}/best.pt')
            else:
                torch.save(model.state_dict(), f'finetuned_models/{model_name}/last.pt')

            if adjust_learning_rate:
                scheduler.step(valid_accuracy)

if __name__ == '__main__':
    main()