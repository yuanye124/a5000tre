import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import FastRGCNConv, RGATConv, GCNConv, GATConv


def info_nce_loss(A, B, temp=0.1, allow_gradient=False):
    labels = torch.cat([torch.arange(A.size(0)) for i in range(1)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    A = F.normalize(A, dim=1)

    if allow_gradient:
        B = F.normalize(B, dim=1)

    else:
        B = F.normalize(B.detach(), dim=1)

    S = torch.matmul(A, B.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    S = S[~mask].view(S.shape[0], -1)

    positives = S[labels.bool()].view(labels.shape[0], -1)
    negatives = S[~labels.bool()].view(S.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    logits = logits / temp
    loss = torch.nn.CrossEntropyLoss()(logits, labels)

    return loss


def supervised_temporal_contrastive_loss(reps, labels, temp=0.1):
    if reps.size(0) <= 1:
        return reps.sum() * 0.0

    reps = F.normalize(reps, dim=1)
    sim = torch.matmul(reps, reps.T) / temp
    batch_size = labels.size(0)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=reps.device)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    valid_mask = positive_mask.any(dim=1)

    if not bool(valid_mask.any().detach().cpu()):
        return reps.sum() * 0.0

    logits = sim - torch.max(sim, dim=1, keepdim=True)[0].detach()
    logits_mask = (~self_mask).float()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_counts = positive_mask.float().sum(dim=1).clamp_min(1.0)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_counts

    return -mean_log_prob_pos[valid_mask].mean()


def _as_probability_distribution(outputs):
    row_sums = outputs.sum(dim=-1)
    is_non_negative = bool((outputs >= 0).all().detach().cpu())
    sums_to_one = bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4))
    if is_non_negative and sums_to_one:
        return outputs

    return F.softmax(outputs, dim=-1)


def inverse_consistency_loss(outputs, epairs, doc_ids, label_reverse_map):
    if label_reverse_map is None or len(label_reverse_map) == 0:
        return outputs.sum() * 0.0

    probs = _as_probability_distribution(outputs)
    num_labels = probs.size(-1)
    reverse_indices = torch.arange(num_labels, device=probs.device)
    for label_id, reverse_label_id in label_reverse_map.items():
        if label_id < num_labels and reverse_label_id < num_labels:
            reverse_indices[label_id] = reverse_label_id

    pair_indices = {}
    losses = []
    for i, (doc_id, epair) in enumerate(zip(doc_ids, epairs)):
        e1, e2 = epair
        reverse_key = (doc_id, e2, e1)
        if reverse_key in pair_indices:
            j = pair_indices[reverse_key]
            losses.append(F.mse_loss(probs[i][reverse_indices], probs[j], reduction='mean'))
            losses.append(F.mse_loss(probs[j][reverse_indices], probs[i], reduction='mean'))
        pair_indices[(doc_id, e1, e2)] = i

    if not losses:
        return outputs.sum() * 0.0

    return torch.stack(losses).mean()




def transitivity_logic_loss(outputs, epairs, doc_ids, transitive_label_ids=None, label_reverse_map=None,
                            confidence_threshold=0.0, max_triplets=2048):
    """
    Differentiable temporal transitivity regularization.

    Conservative first version:
    - only applies to labels listed in transitive_label_ids, usually before/after;
    - uses probabilities from the current batch;
    - if both (A,B) and (B,C) strongly support relation r, then (A,C) should also support r;
    - supports reversed pairs by using label_reverse_map to construct reverse-direction probabilities.

    Loss for each valid triple and relation r:
        ReLU(P_r(A,B) * P_r(B,C) - P_r(A,C))
    """
    if transitive_label_ids is None or len(transitive_label_ids) == 0 or outputs.size(0) <= 2:
        return outputs.sum() * 0.0

    probs = _as_probability_distribution(outputs)
    num_labels = probs.size(-1)

    valid_label_ids = []
    for label_id in transitive_label_ids:
        label_id = int(label_id)
        if 0 <= label_id < num_labels:
            valid_label_ids.append(label_id)
    if len(valid_label_ids) == 0:
        return outputs.sum() * 0.0

    reverse_indices = None
    if label_reverse_map is not None and len(label_reverse_map) > 0:
        reverse_indices = torch.arange(num_labels, device=probs.device)
        for label_id, reverse_label_id in label_reverse_map.items():
            label_id = int(label_id)
            reverse_label_id = int(reverse_label_id)
            if 0 <= label_id < num_labels and 0 <= reverse_label_id < num_labels:
                reverse_indices[label_id] = reverse_label_id

    pair_probs = {}
    outgoing = {}
    for i, (doc_id, epair) in enumerate(zip(doc_ids, epairs)):
        e1, e2 = epair
        key = (doc_id, e1, e2)
        pair_probs[key] = probs[i]
        outgoing.setdefault((doc_id, e1), []).append((e2, probs[i]))

        # Add a soft reverse-direction view when label_reverse_map is available.
        # Example: P_before(e2,e1) is taken from P_after(e1,e2).
        if reverse_indices is not None:
            rev_key = (doc_id, e2, e1)
            rev_probs = probs[i][reverse_indices]
            pair_probs[rev_key] = rev_probs
            outgoing.setdefault((doc_id, e2), []).append((e1, rev_probs))

    losses = []
    triplet_count = 0
    for (doc_id, a), b_prob_list in outgoing.items():
        for b, p_ab in b_prob_list:
            for c, p_bc in outgoing.get((doc_id, b), []):
                if a == c:
                    continue
                p_ac = pair_probs.get((doc_id, a, c), None)
                if p_ac is None:
                    continue

                for label_id in valid_label_ids:
                    premise_left = p_ab[label_id]
                    premise_right = p_bc[label_id]
                    if confidence_threshold > 0.0:
                        if premise_left.detach() < confidence_threshold or premise_right.detach() < confidence_threshold:
                            continue
                    premise = premise_left * premise_right
                    losses.append(F.relu(premise - p_ac[label_id]))
                    triplet_count += 1
                    if max_triplets is not None and triplet_count >= max_triplets:
                        break
                if max_triplets is not None and triplet_count >= max_triplets:
                    break
            if max_triplets is not None and triplet_count >= max_triplets:
                break
        if max_triplets is not None and triplet_count >= max_triplets:
            break

    if len(losses) == 0:
        return outputs.sum() * 0.0

    return torch.stack(losses).mean()

def get_gcn(gcn_type, input_dim, output_dim, num_edge_type):
    if gcn_type == "gcn":
        gcn = GCNConv(input_dim, output_dim)

    if gcn_type == "gat":
        gcn = GATConv(input_dim, output_dim)

    if gcn_type == "rgcn":
        gcn = FastRGCNConv(input_dim, output_dim, num_edge_type)

    if gcn_type == "rgat":
        gcn = RGATConv(input_dim, output_dim, num_edge_type)

    return gcn


class RGCN_v1(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_edge_type, gcn_type="rgcn"):
        super(RGCN_v1, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.num_edge_type = num_edge_type
        self.rgcn_layers = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        self.input_proj = nn.Linear(self.num_layers * self.hidden_size, self.hidden_size)
        self.gcn_type = gcn_type

        for i in range(self.num_layers):
            if i == 0:
                self.rgcn_layers.append(get_gcn(self.gcn_type, self.input_dim, self.hidden_size, self.num_edge_type))
            else:
                self.rgcn_layers.append(get_gcn(self.gcn_type, self.hidden_size, self.hidden_size, self.num_edge_type))

    def forward(self, graph):
        node_feat = graph.node_feat.cuda()
        edge_index = graph.edge_index.cuda()
        edge_type = graph.edge_type.cuda()
        graph_reps = []
        cur_graph_rep = node_feat

        for i in range(self.num_layers):
            if self.gcn_type == "gcn":
                h = self.rgcn_layers[i](cur_graph_rep, edge_index)
            else:
                h = self.rgcn_layers[i](cur_graph_rep, edge_index, edge_type)

            cur_graph_rep = self.relu(h)
            graph_reps.append(cur_graph_rep)

        final_graph_rep = self.input_proj(torch.cat(graph_reps, dim=-1))

        return final_graph_rep, graph_reps


class BertTemporalOrdering(nn.Module):
    def __init__(self, encoder, config, use_evidence_attention=False, evidence_boost=0.1,
                 middle_evidence_boost=0.05):
        super(BertTemporalOrdering, self).__init__()
        self.encoder = encoder
        self.config = config
        self.use_evidence_attention = use_evidence_attention
        self.evidence_boost = evidence_boost
        self.middle_evidence_boost = middle_evidence_boost
        self.attention_proj = nn.ModuleList()
        self.attention_queries = []

        for i in range(5):
            self.attention_proj.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
            self.attention_queries.append(torch.rand(1, self.config.hidden_size, requires_grad=True))

        self.attention_queries = nn.Parameter(torch.vstack(self.attention_queries))

    def word_attention(self, words, index):
        query = self.attention_queries[index, :].unsqueeze(0)
        projected_words = nn.Tanh()(self.attention_proj[index](words))
        alphas = torch.matmul(query, projected_words.permute(1, 0))
        max_value = torch.max(alphas.detach())
        alphas = alphas - max_value
        alphas = nn.Softmax(dim=1)(alphas)
        att_rep = torch.sum(words * alphas.permute(1, 0).repeat(1, self.config.hidden_size), dim=0).unsqueeze(0)

        return att_rep

    def _apply_evidence_attention(self, sub_reps, cue_features):
        if (not self.use_evidence_attention) or cue_features is None:
            return sub_reps

        cue_features = cue_features.cuda().float()
        segment_weights = 1.0 + self.evidence_boost * cue_features
        segment_weights[:, 2] = segment_weights[:, 2] + self.middle_evidence_boost * cue_features[:, 2]
        segment_weights = torch.clamp(segment_weights, min=1.0, max=1.0 + self.evidence_boost + self.middle_evidence_boost)

        weighted_sub_reps = []
        for i, rep in enumerate(sub_reps):
            weighted_sub_reps.append(rep * segment_weights[:, i:i + 1])
        return weighted_sub_reps

    def forward(self, batch):
        if len(batch) == 3:
            tokenized_inputs, indices, cue_features = batch
        else:
            tokenized_inputs, indices = batch
            cue_features = None

        tokenized_inputs = {x: y.cuda() for x, y in tokenized_inputs.items() if x != 'labels'}
        outputs = self.encoder(**tokenized_inputs, output_hidden_states=True)
        full_emb = outputs[1]
        last_hidden_states = outputs[2][-1]
        sub_reps = [[], [], [], [], []]

        for i, example in enumerate(indices):
            for j, seq in enumerate(example):
                start, end = seq
                cur_words = last_hidden_states[i, start:end, :]
                mean_pool_rep = torch.mean(cur_words, dim=0).unsqueeze(0) if start != end else torch.zeros(
                    (1, self.config.hidden_size)).cuda()
                cur_att_rep = self.word_attention(cur_words, j) if start != end else torch.zeros(
                    (1, self.config.hidden_size)).cuda()
                sub_reps[j].append(torch.cat([mean_pool_rep, cur_att_rep], dim=1))

        sub_reps = [torch.vstack(x) for x in sub_reps]
        sub_reps = self._apply_evidence_attention(sub_reps, cue_features)
        sub_reps.append(full_emb)
        final_reps = torch.cat(sub_reps, dim=1)

        return final_reps, full_emb


class MulCo(nn.Module):
    def __init__(self, syntax_gcn, temporal_gcn, context_encoder, classifier, dropout, k_hops, temp=1,
                 bert_classifier=None, gcn_classifier=None, multi_scale=False, alpha_cl=1.0, beta_tcl=0.2,
                 gamma_inv=0.1, label_reverse_map=None, logic_loss_weight=0.0,
                 transitive_label_ids=None, logic_conf_threshold=0.0):
        super(MulCo, self).__init__()
        self.syntax_gcn = syntax_gcn
        self.temporal_gcn = temporal_gcn
        self.context_encoder = context_encoder
        self.classifier = classifier
        self.dropout = nn.Dropout(dropout)
        self.temp = temp
        self.bert_classifier = bert_classifier
        self.gcn_classifier = gcn_classifier
        self.clf_loss_fn = nn.CrossEntropyLoss()
        self.context_encoder_temp = None
        self.syntax_gcn_temp = None
        self.temporal_gcn_temp = None
        self.multi_scale = multi_scale
        self.alpha_cl = alpha_cl
        self.beta_tcl = beta_tcl
        self.gamma_inv = gamma_inv
        self.label_reverse_map = label_reverse_map
        self.logic_loss_weight = logic_loss_weight
        self.transitive_label_ids = transitive_label_ids if transitive_label_ids is not None else []
        self.logic_conf_threshold = logic_conf_threshold

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.syntax_gcn.output_size, nhead=1,
                                                   batch_first=True).cuda()
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1).cuda()
        self.k_hops = k_hops
        gcn_head_size = self.syntax_gcn.hidden_size
        self.gcn_head = nn.Sequential(nn.Linear(gcn_head_size, 2048, bias=False), nn.Linear(2048, 2048, bias=False),
                                      nn.ReLU()).cuda()
        self.temporal_cl_head = nn.Sequential(nn.Linear(gcn_head_size, gcn_head_size, bias=False),
                                              nn.ReLU(),
                                              nn.Linear(gcn_head_size, gcn_head_size, bias=False),
                                              nn.ReLU()).cuda()
        self.bert_head = nn.Sequential(nn.Linear(768, 2048, bias=False), nn.Linear(2048, 2048, bias=False),
                                       nn.ReLU()).cuda()

    def _k_hop_nodes(self, node_idx, edge_index):
        return torch_geometric.utils.k_hop_subgraph(node_idx, self.k_hops, edge_index)[0]

    def _encode_subgraphs(self, reps_one, reps_two):
        return torch.mean(self.transformer_encoder(torch.vstack([reps_one, reps_two]).unsqueeze(0)), dim=1)

    def _encode_pair_multiscale(self, syntax_reps, temporal_reps, e1_syntax_node_idx, e2_syntax_node_idx,
                                e1_temporal_node_idx, e2_temporal_node_idx, syntax_edge_index, temporal_edge_index):
        pair_scale_embs = []

        for syntax_graph_emb in syntax_reps:
            e1_syntax_subgraph = syntax_graph_emb[self._k_hop_nodes(e1_syntax_node_idx, syntax_edge_index)]
            e2_syntax_subgraph = syntax_graph_emb[self._k_hop_nodes(e2_syntax_node_idx, syntax_edge_index)]
            pair_scale_embs.append(self._encode_subgraphs(e1_syntax_subgraph, e2_syntax_subgraph))

        for temporal_graph_emb in temporal_reps:
            e1_temporal_subgraph = temporal_graph_emb[self._k_hop_nodes(e1_temporal_node_idx, temporal_edge_index)]
            e2_temporal_subgraph = temporal_graph_emb[self._k_hop_nodes(e2_temporal_node_idx, temporal_edge_index)]
            pair_scale_embs.append(self._encode_subgraphs(e1_temporal_subgraph, e2_temporal_subgraph))

        return torch.mean(self.transformer_encoder(torch.vstack(pair_scale_embs).unsqueeze(0)), dim=1)

    def _encode_pair_single_scale(self, syntax_graph_emb, temporal_graph_emb, e1_syntax_node_idx, e2_syntax_node_idx,
                                  e1_temporal_node_idx, e2_temporal_node_idx, syntax_edge_index, temporal_edge_index):
        e1_syntax_subgraph = syntax_graph_emb[self._k_hop_nodes(e1_syntax_node_idx, syntax_edge_index)]
        e2_syntax_subgraph = syntax_graph_emb[self._k_hop_nodes(e2_syntax_node_idx, syntax_edge_index)]
        e1_temporal_subgraph = temporal_graph_emb[self._k_hop_nodes(e1_temporal_node_idx, temporal_edge_index)]
        e2_temporal_subgraph = temporal_graph_emb[self._k_hop_nodes(e2_temporal_node_idx, temporal_edge_index)]

        return torch.mean(
            self.transformer_encoder(
                torch.vstack([
                    e1_syntax_subgraph,
                    e2_syntax_subgraph,
                    e1_temporal_subgraph,
                    e2_temporal_subgraph
                ]).unsqueeze(0)
            ),
            dim=1
        )

    def forward(self, batch):
        # Backward-compatible batch parsing.
        # The optional LLM refinement branch appends prompt payloads after cue_features during testing.
        # Model computation only needs the first 8 items, so any extra items are safely ignored here.
        if len(batch) >= 8:
            tokenized_inputs, token_index, indices, doc_ids, epairs, syntax_graphs, temporal_graphs, cue_features = batch[:8]
        else:
            tokenized_inputs, token_index, indices, doc_ids, epairs, syntax_graphs, temporal_graphs = batch
            cue_features = None
        syntax_graph_emb_dict = {}
        multi_syntax_graph_emb_dict = {}
        temporal_graph_emb_dict = {}
        multi_temporal_graph_emb_dict = {}
        e1_embs = []
        e2_embs = []
        g_embs = []

        concat_reps = []
        bert_outputs = None
        gcn_outputs = None

        if self.syntax_gcn:
            for i, doc_id in enumerate(doc_ids):
                if doc_id not in syntax_graph_emb_dict:
                    syntax_graph = syntax_graphs[i]
                    syntax_graph_emb, syntax_graph_reps = self.syntax_gcn(syntax_graph)
                    syntax_graph_emb_dict[doc_id] = syntax_graph_emb
                    multi_syntax_graph_emb_dict[doc_id] = syntax_graph_reps

                if doc_id not in temporal_graph_emb_dict:
                    temporal_graph = temporal_graphs[i]
                    temporal_graph_emb, temporal_graph_reps = self.temporal_gcn(temporal_graph)
                    temporal_graph_emb_dict[doc_id] = temporal_graph_emb
                    multi_temporal_graph_emb_dict[doc_id] = temporal_graph_reps

                e1, e2 = epairs[i]
                e1_syntax_node_idx = temporal_graphs[i].event_to_word_idx[e1]
                e1_temporal_node_idx = temporal_graphs[i].node_idx[e1]
                e2_syntax_node_idx = temporal_graphs[i].event_to_word_idx[e2]
                e2_temporal_node_idx = temporal_graphs[i].node_idx[e2]
                e1_syntax_subgraphs = syntax_graph_emb_dict[doc_id][
                    self._k_hop_nodes(e1_syntax_node_idx, syntax_graphs[i].edge_index)]
                e1_temporal_subgraphs = temporal_graph_emb_dict[doc_id][
                    self._k_hop_nodes(e1_temporal_node_idx, temporal_graphs[i].edge_index)]
                e1_emb = self._encode_subgraphs(e1_syntax_subgraphs, e1_temporal_subgraphs)
                e2_syntax_subgraphs = syntax_graph_emb_dict[doc_id][
                    self._k_hop_nodes(e2_syntax_node_idx, syntax_graphs[i].edge_index)]
                e2_temporal_subgraphs = temporal_graph_emb_dict[doc_id][
                    self._k_hop_nodes(e2_temporal_node_idx, temporal_graphs[i].edge_index)]
                e2_emb = self._encode_subgraphs(e2_syntax_subgraphs, e2_temporal_subgraphs)
                e1_embs.append(e1_emb)
                e2_embs.append(e2_emb)

                if self.multi_scale:
                    g_emb = self._encode_pair_multiscale(
                        multi_syntax_graph_emb_dict[doc_id],
                        multi_temporal_graph_emb_dict[doc_id],
                        e1_syntax_node_idx,
                        e2_syntax_node_idx,
                        e1_temporal_node_idx,
                        e2_temporal_node_idx,
                        syntax_graphs[i].edge_index,
                        temporal_graphs[i].edge_index
                    )
                else:
                    g_emb = self._encode_pair_single_scale(
                        syntax_graph_emb_dict[doc_id],
                        temporal_graph_emb_dict[doc_id],
                        e1_syntax_node_idx,
                        e2_syntax_node_idx,
                        e1_temporal_node_idx,
                        e2_temporal_node_idx,
                        syntax_graphs[i].edge_index,
                        temporal_graphs[i].edge_index
                    )
                g_embs.append(g_emb)

            e1_embs = torch.vstack(e1_embs)
            e2_embs = torch.vstack(e2_embs)
            g_embs = torch.vstack(g_embs)
            gcn_reps = torch.cat([e1_embs, e2_embs], dim=-1)
            concat_reps.append(gcn_reps)
            gcn_outputs = self.gcn_classifier(gcn_reps)

        if self.context_encoder:
            bert_reps, bert_embs = self.context_encoder((tokenized_inputs, indices, cue_features))
            concat_reps.append(bert_reps)
            bert_outputs = self.bert_classifier(bert_reps)

        final_output = self.classifier(self.dropout(torch.cat(concat_reps, dim=-1)))
        z_gcn_reps = self.gcn_head(g_embs)
        z_bert_reps = self.bert_head(bert_embs)

        loss = 0.0
        if "labels" in tokenized_inputs:
            labels = tokenized_inputs['labels'].cuda()
            bert_graph_cl_loss = (info_nce_loss(z_gcn_reps, z_bert_reps, self.temp) + info_nce_loss(z_bert_reps,
                                                                                                    z_gcn_reps,
                                                                                                    self.temp)) / 2
            temporal_contrastive_loss = supervised_temporal_contrastive_loss(gcn_reps, labels, self.temp)
            inverse_loss = inverse_consistency_loss(final_output, epairs, doc_ids, self.label_reverse_map)
            logic_loss = transitivity_logic_loss(
                final_output,
                epairs,
                doc_ids,
                self.transitive_label_ids,
                self.label_reverse_map,
                self.logic_conf_threshold
            )
            clf_loss = self.clf_loss_fn(final_output, labels)
            loss = clf_loss + self.alpha_cl * bert_graph_cl_loss + self.beta_tcl * temporal_contrastive_loss + \
                self.gamma_inv * inverse_loss + self.logic_loss_weight * logic_loss

        if getattr(self, "return_aux", False):
            aux_outputs = {
                "bert_outputs": bert_outputs,
                "gcn_outputs": gcn_outputs,
                "final_output": final_output,
            }
            return loss, final_output, aux_outputs

        return loss, final_output
