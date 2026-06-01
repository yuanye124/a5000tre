import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class BaseTrainer:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def train(self, model, train_data, dev_data, epochs, path, rel_dict, distance_flags, accumulation_steps):
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        best_dev_loss = float('inf')
        training_details = []
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
                    if (step+1) == num_batches:
                        loss = loss / ((step+1) % accumulation_steps)

                    else:
                        loss = loss / accumulation_steps

                    total_loss += loss.item()
                    loss.backward()

                    if ((step+1) % accumulation_steps == 0) or ((step+1) == num_batches):
                        optimizer.step() 
                        optimizer.zero_grad()

                else:
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step() 
                    optimizer.zero_grad()
            
            total_loss /= num_batches     
            dev_loss = self.validate(model, dev_data)
            val_str = "Validation Loss at epoch {}: {}".format(epoch, dev_loss)
            print(val_str)

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                torch.save(model.state_dict(), os.path.join(path, 'best_model.pt'))

    @torch.no_grad()
    def validate(self, model, dev_data):
        model.eval()
        val_loss = 0.0
        num_samples = len(dev_data)
        prec_num = 0.0
        prec_denom = 0.0
        for sample in dev_data:
            loss, outputs = model(sample)
            prec_denom += outputs.size()[0]
            chosen_class = torch.argmax(outputs, dim=-1)
            chosen_class = chosen_class.detach().cpu().numpy().tolist()
            gold_labels = sample[0]['labels'].numpy().tolist()
            for x,y in zip(chosen_class, gold_labels):
                if x == y:
                    prec_num += 1
            val_loss += loss.item()
        print('Precision: {}'.format(prec_num/prec_denom))
        return val_loss / num_samples


    @torch.no_grad()
    def test_with_confidence(self, model, test_data, path=None):
        """
        Run test and return both hard predictions and confidence records.
        This log-aux version also records BERT-only and GNN-only branch predictions
        when the model supports return_aux.
        """
        if path:
            model.load_state_dict(torch.load(os.path.join(path, 'best_model.pt')))

        model.eval()
        predictions = {}
        confidence_records = []

        old_return_aux = getattr(model, 'return_aux', False)
        model.return_aux = True

        def _to_probs(outputs):
            if outputs is None:
                return None
            row_sums = outputs.sum(dim=-1)
            is_non_negative = bool((outputs >= 0).all().detach().cpu())
            sums_to_one = bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4))
            return outputs if (is_non_negative and sums_to_one) else torch.softmax(outputs, dim=-1)

        def _top_info(prob_row):
            if prob_row is None:
                return None
            vals, idxs = torch.topk(prob_row, k=min(2, prob_row.size(-1)), dim=-1)
            vals = vals.detach().cpu().numpy().tolist()
            idxs = idxs.detach().cpu().numpy().tolist()
            return {
                'top1': int(idxs[0]),
                'top1_prob': float(vals[0]),
                'top2': int(idxs[1]) if len(idxs) > 1 else None,
                'top2_prob': float(vals[1]) if len(vals) > 1 else 0.0,
                'probs': prob_row.detach().cpu().numpy().tolist(),
            }

        for sample in test_data:
            doc_ids = sample[3]
            epairs = sample[4]
            payloads = sample[8] if len(sample) > 8 else [None for _ in epairs]
            model_out = model(sample)
            if len(model_out) == 3:
                _, outputs, aux_outputs = model_out
            else:
                _, outputs = model_out
                aux_outputs = {}

            probs = _to_probs(outputs)
            bert_probs = _to_probs(aux_outputs.get('bert_outputs'))
            gcn_probs = _to_probs(aux_outputs.get('gcn_outputs'))

            top_values, top_indices = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1)
            chosen_class = top_indices[:, 0].detach().cpu().numpy().tolist()
            top_values_cpu = top_values.detach().cpu().numpy().tolist()
            probs_cpu = probs.detach().cpu().numpy().tolist()

            for i, pred in enumerate(chosen_class):
                if doc_ids[i] not in predictions:
                    predictions[doc_ids[i]] = {}
                predictions[doc_ids[i]][epairs[i]] = pred

                top1 = float(top_values_cpu[i][0])
                top2 = float(top_values_cpu[i][1]) if len(top_values_cpu[i]) > 1 else 0.0
                record = {
                    'doc_id': doc_ids[i],
                    'epair': epairs[i],
                    'pred': pred,
                    'confidence': top1,
                    'margin': top1 - top2,
                    'probs': probs_cpu[i],
                    'payload': payloads[i] if i < len(payloads) else None,
                }
                if bert_probs is not None:
                    record['bert_branch'] = _top_info(bert_probs[i])
                if gcn_probs is not None:
                    record['gcn_branch'] = _top_info(gcn_probs[i])
                confidence_records.append(record)

        model.return_aux = old_return_aux
        return predictions, confidence_records

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
