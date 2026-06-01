import os
import random
import torch
import torch.optim as optim
from tqdm import tqdm


class BaseTrainer:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def train(self, model, train_data, dev_data, epochs, path, rel_dict, distance_flags, accumulation_steps):
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        best_dev_f1 = -1.0
        best_dev_loss = float('inf')
        print('Storing models at {}'.format(path))

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            num_batches = len(train_data)
            random.shuffle(train_data)
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_data, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
                loss, _ = model(batch)

                if accumulation_steps != 0:
                    if (step + 1) == num_batches:
                        remainder = (step + 1) % accumulation_steps
                        loss = loss / (remainder if remainder != 0 else accumulation_steps)
                    else:
                        loss = loss / accumulation_steps

                    total_loss += loss.item()
                    loss.backward()

                    if ((step + 1) % accumulation_steps == 0) or ((step + 1) == num_batches):
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss /= max(num_batches, 1)
            dev_loss, dev_f1 = self.validate(model, dev_data)
            print('Validation Loss at epoch {}: {}'.format(epoch, dev_loss))
            print('Validation F1 at epoch {}: {}'.format(epoch, dev_f1))

            # Select checkpoint by dev F1 first. If F1 ties, keep lower dev loss.
            if (dev_f1 > best_dev_f1) or (dev_f1 == best_dev_f1 and dev_loss < best_dev_loss):
                best_dev_f1 = dev_f1
                best_dev_loss = dev_loss
                torch.save(model.state_dict(), os.path.join(path, 'best_model.pt'))
                print('Saved best model at epoch {} with dev F1 {:.6f}'.format(epoch, dev_f1))

    @torch.no_grad()
    def validate(self, model, dev_data):
        model.eval()
        val_loss = 0.0
        num_samples = len(dev_data)
        correct_num = 0.0
        pred_num = 0.0
        gold_num = 0.0

        for sample in dev_data:
            loss, outputs = model(sample)
            chosen_class = torch.argmax(outputs, dim=-1)
            chosen_class = chosen_class.detach().cpu().numpy().tolist()
            gold_labels = sample[0]['labels'].numpy().tolist()

            for pred, gold in zip(chosen_class, gold_labels):
                pred_num += 1
                gold_num += 1
                if pred == gold:
                    correct_num += 1

            val_loss += loss.item()

        precision = correct_num / pred_num if pred_num != 0.0 else 0.0
        recall = correct_num / gold_num if gold_num != 0.0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0.0 else 0.0
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F1: {}'.format(f1))
        return val_loss / max(num_samples, 1), f1

    @torch.no_grad()
    def test(self, model, test_data, path=None):
        if path:
            model.load_state_dict(torch.load(os.path.join(path, 'best_model.pt')))

        model.eval()
        predictions = {}
        for sample in test_data:
            doc_ids = sample[3]
            epairs = sample[4]
            _, outputs = model(sample)
            chosen_class = torch.argmax(outputs, dim=-1)
            chosen_class = chosen_class.detach().cpu().numpy().tolist()
            for i, pred in enumerate(chosen_class):
                if doc_ids[i] not in predictions:
                    predictions[doc_ids[i]] = {}
                predictions[doc_ids[i]][epairs[i]] = pred
        return predictions
