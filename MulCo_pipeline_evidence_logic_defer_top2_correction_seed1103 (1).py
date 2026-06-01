import argparse
import os
import pickle
import json
import networkx
from datetime import datetime
from tqdm import tqdm
import re

parser = argparse.ArgumentParser()
parser.add_argument('--doc_file', type=str, action='store', required=True,
                    help='Path to file containing processed documents')
parser.add_argument('--gold_pairs', type=str, action='store', required=True,
                    help='Path to file containing gold event pairs')
parser.add_argument('--test_pairs', type=str, action='store', required=True,
                    help='Path to file containing test event pairs')
parser.add_argument('--out_dir', type=str, action='store', required=True, help='Path to store model outputs')
parser.add_argument('--event_map', type=str, action='store', default='data/event_map.pkl',
                    help='Path to file containing e-ID to ei-ID mapping')
parser.add_argument('--bert_model', type=str, action='store', default='bert', help='BERT model to use')
parser.add_argument('--bert_encoder_type', type=str, action='store', default='neighbor',
                    help='BERT encoding type to use')
parser.add_argument('--learning_rate', type=float, action='store', default=1e-5, help='Learning rate')
parser.add_argument('--dropout', type=float, action='store', default=0.5, help='Learning rate')
parser.add_argument('--batch_size', type=int, action='store', default=16, help='Batch size')
parser.add_argument('--accumulation_steps', type=int, action='store', default=2, help='Accumulation steps')
parser.add_argument('--epochs', type=int, action='store', default=10, help='Number of epochs')
parser.add_argument('--syntax_file', type=str, action='store', default='syntax_graph.pkl',
                    help='Path to file containing syntax graphs')
parser.add_argument('--temporal_file', type=str, action='store', default='time_graph.pkl',
                    help='Path to file containing temporal graphs')
parser.add_argument('--gcn_hidden_size', type=int, action='store', default=64, help='Hidden size for syntax GCN module')
parser.add_argument('--gcn_num_layers', type=int, action='store', default=1,
                    help='Number of layers for syntax GCN module')
parser.add_argument('--dist_cutoff', type=int, action='store', default=2,
                    help='Distance at which events should be considered short-distance')
parser.add_argument('--dist_type', type=str, action='store', default="mix", help='mix/long/short training only')
parser.add_argument('--bert_or_gcn', type=str, action='store', default="both", help='bert/gcn/both for temporal model')
parser.add_argument('--gcn_type', type=str, action='store', default="rgcn", help='type of gcn used in the model')
parser.add_argument('--seed', type=int, action='store', default=1103, help='Random seed for reproducibility')
parser.add_argument('--k_hops', type=int, action='store', default=1,
                    help='Number of hop for k-hop subgraph for each node')
parser.add_argument('--temp', type=float, action='store', default=0.1, help='Temperature for contrastive objective')
parser.add_argument('--alpha_cl', type=float, action='store', default=1.0,
                    help='Weight for BERT-Graph contrastive loss')
parser.add_argument('--beta_tcl', type=float, action='store', default=0.2,
                    help='Weight for supervised temporal contrastive loss')
parser.add_argument('--gamma_inv', type=float, action='store', default=0.1,
                    help='Weight for inverse temporal consistency loss')
parser.add_argument('--logic_loss_weight', type=float, action='store', default=0.0,
                    help='Weight for transitivity temporal logic loss')
parser.add_argument('--logic_conf_threshold', type=float, action='store', default=0.0,
                    help='Confidence threshold for using premise edges in transitivity logic loss')
parser.add_argument('--use_evidence_attention', action='store_true',
                    help='Enable lightweight temporal cue evidence weighting in BERT segment representations')
parser.add_argument('--evidence_boost', type=float, action='store', default=0.1,
                    help='Boost factor for segments containing temporal cue words')
parser.add_argument('--middle_evidence_boost', type=float, action='store', default=0.05,
                    help='Extra boost for the middle segment when it contains temporal cue words')
parser.add_argument('--use_llm_refine', action='store_true',
                    help='Use a small LLM to re-classify low-confidence test event pairs after MulCo prediction')
parser.add_argument('--llm_model_name', type=str, action='store', default='google/flan-t5-small',
                    help='Small text-to-text model name or local path for low-confidence LLM refinement')
parser.add_argument('--llm_conf_threshold', type=float, action='store', default=0.65,
                    help='Refine samples whose top prediction confidence is lower than this threshold')
parser.add_argument('--llm_margin_threshold', type=float, action='store', default=0.15,
                    help='Refine samples whose top1-top2 probability margin is lower than this threshold')
parser.add_argument('--llm_max_cases', type=int, action='store', default=200,
                    help='Maximum number of low-confidence test pairs refined by LLM; use 0 for all candidates')
parser.add_argument('--llm_device', type=str, action='store', default='cpu', choices=['cpu', 'cuda'],
                    help='Device for the small LLM. CPU is safer for limited GPU memory')
parser.add_argument('--llm_max_input_length', type=int, action='store', default=512,
                    help='Maximum input token length for the LLM prompt')
parser.add_argument('--llm_max_new_tokens', type=int, action='store', default=8,
                    help='Maximum generated tokens for the LLM label answer')
parser.add_argument('--llm_refine_mode', type=str, action='store', default='top2_judge',
                    choices=['free', 'top2_judge', 'verify_then_top2', 'log_guided_ksu', 'correction_mlp'],
                    help='Refinement mode. correction_mlp does not load an LLM; it uses a dev-trained top-2 correction network to decide KEEP/SWITCH.')
parser.add_argument('--llm_replace_margin_threshold', type=float, action='store', default=0.08,
                    help='In top2_judge mode, replace top1 by top2 only when top1-top2 margin is not larger than this value')
parser.add_argument('--allow_llm_vague_replace', action='store_true',
                    help='Allow LLM to replace a non-vague prediction with vague. Default is disabled for safety')
parser.add_argument('--llm_top2_score_method', type=str, action='store', default='token_score',
                    choices=['token_score', 'generate'],
                    help='For top2_judge mode, use token_score to compare next-token scores for 1/2, or generate to parse generated text')
parser.add_argument('--llm_double_order_score', action='store_true',
                    help='Use two prompt orders to reduce option-position bias: top1/top2 and top2/top1 are both scored and averaged')
parser.add_argument('--llm_candidate_filter', type=str, action='store', default='or',
                    choices=['or', 'and', 'margin'],
                    help='How to select LLM candidates: or=confidence<thr OR margin<thr; and=both; margin=margin only')
parser.add_argument('--llm_use_context_evidence', action='store_true',
                    help='Use full context plus focused evidence snippets in the LLM top2 judge prompt')
parser.add_argument('--llm_use_verifier', action='store_true',
                    help='Before replacing with top2, verify whether top1 is unsupported and top2 is better supported')
parser.add_argument('--llm_score_gap_threshold', type=float, action='store', default=0.0,
                    help='Minimum LLM option score gap required before allowing top2 replacement')
parser.add_argument('--llm_verify_delta', type=float, action='store', default=0.6,
                    help='Minimum verifier score advantage of top2 over top1 required for replacement')
parser.add_argument('--llm_verify_top1_threshold', type=float, action='store', default=0.0,
                    help='Replacement requires top1 verifier support score not larger than this threshold')
parser.add_argument('--llm_switch_delta', type=float, action='store', default=0.8,
                    help='In log_guided_ksu mode, switch only when score(SWITCH)-score(KEEP) exceeds this threshold')
parser.add_argument('--llm_branch_disagreement_only', action='store_true',
                    help='In log_guided_ksu mode, send only candidates where BERT/GNN/final branches disagree or one branch supports top2')
parser.add_argument('--llm_audit_only', action='store_true',
                    help='Run LLM audit and save records but never change final predictions')
parser.add_argument('--llm_auto_threshold_from_dev', action='store_true',
                    help='Calibrate LLM candidate-selection threshold on the dev set before refining test samples')
parser.add_argument('--llm_dev_target_error_recall', type=float, action='store', default=0.6,
                    help='Target recall of dev-set errors when selecting LLM candidates by margin')
parser.add_argument('--llm_dev_max_candidate_rate', type=float, action='store', default=0.12,
                    help='Maximum fraction of dev samples allowed to enter LLM candidate region during calibration')
parser.add_argument('--llm_dev_min_candidates', type=int, action='store', default=1,
                    help='Minimum candidate budget used during dev-set threshold calibration')
parser.add_argument('--llm_dev_max_margin_threshold', type=float, action='store', default=0.20,
                    help='Upper bound for the calibrated LLM margin threshold')
parser.add_argument('--llm_dev_force_margin_filter', action='store_true',
                    help='After dev calibration, force candidate filtering to use margin-only mode')

parser.add_argument('--llm_candidate_strategy', type=str, action='store', default='threshold',
                    choices=['threshold', 'risk_rank', 'defer_mlp'],
                    help='LLM candidate selection strategy. threshold uses confidence/margin thresholds; risk_rank ranks samples by dev-calibrated error risk and selects top-K; defer_mlp trains a small deferral network on dev correctness to decide which samples should be sent to LLM')
parser.add_argument('--llm_candidate_target_count', type=int, action='store', default=0,
                    help='Target number of test samples sent to LLM under risk_rank. 0 means use dev-calibrated target rate')
parser.add_argument('--llm_candidate_max_count', type=int, action='store', default=30,
                    help='Maximum number of test samples sent to LLM under risk_rank')
parser.add_argument('--llm_risk_auto_target_count', action='store_true',
                    help='Use dev-set risk ranking to automatically determine the target candidate rate')
parser.add_argument('--llm_dev_min_candidate_error_precision', type=float, action='store', default=0.40,
                    help='When auto-selecting candidate count on dev, require at least this error precision among selected risky samples')
parser.add_argument('--llm_min_risk_for_candidate', type=float, action='store', default=0.0,
                    help='Minimum risk score required to enter LLM candidate set under risk_rank. 0 disables this filter')
parser.add_argument('--llm_min_risk_for_replace', type=float, action='store', default=0.0,
                    help='Minimum risk score required before a SWITCH decision can replace top1. 0 disables this safety check')
parser.add_argument('--llm_risk_margin_cap', type=float, action='store', default=0.20,
                    help='Margin value at which margin-risk becomes zero')
parser.add_argument('--llm_margin_weight', type=float, action='store', default=0.35,
                    help='Risk score weight for top1-top2 margin uncertainty')
parser.add_argument('--llm_conf_weight', type=float, action='store', default=0.20,
                    help='Risk score weight for low confidence')
parser.add_argument('--llm_entropy_weight', type=float, action='store', default=0.25,
                    help='Risk score weight for prediction entropy')
parser.add_argument('--llm_profile_weight', type=float, action='store', default=0.20,
                    help='Risk score weight for dev-set error profile')
parser.add_argument('--llm_dev_error_profile_bins', type=str, action='store', default='0.05,0.08,0.12,0.20',
                    help='Comma-separated margin bin edges for dev-calibrated risk profiles')
parser.add_argument('--llm_defer_epochs', type=int, action='store', default=250,
                    help='Training epochs for the small dev-trained deferral MLP')
parser.add_argument('--llm_defer_hidden_size', type=int, action='store', default=32,
                    help='Hidden size for the deferral MLP')
parser.add_argument('--llm_defer_lr', type=float, action='store', default=1e-3,
                    help='Learning rate for the deferral MLP')
parser.add_argument('--llm_defer_dropout', type=float, action='store', default=0.15,
                    help='Dropout for the deferral MLP')
parser.add_argument('--llm_defer_min_error_precision', type=float, action='store', default=0.40,
                    help='On dev, choose a defer candidate budget whose wrong-sample precision is at least this value when possible')
parser.add_argument('--llm_defer_max_candidate_rate', type=float, action='store', default=0.20,
                    help='Maximum dev/test fraction allowed to be selected by the deferral MLP')
parser.add_argument('--llm_defer_min_candidates', type=int, action='store', default=3,
                    help='Minimum candidate budget for deferral MLP selection')
parser.add_argument('--llm_defer_use_margin_replace_gate', action='store_true',
                    help='If enabled, still require margin <= llm_replace_margin_threshold before applying a SWITCH; otherwise defer MLP controls handoff and LLM switch_delta controls replacement')

parser.add_argument('--correction_epochs', type=int, action='store', default=300,
                    help='Training epochs for the top-2 correction MLP')
parser.add_argument('--correction_hidden_size', type=int, action='store', default=32,
                    help='Hidden size for the top-2 correction MLP')
parser.add_argument('--correction_lr', type=float, action='store', default=1e-3,
                    help='Learning rate for the top-2 correction MLP')
parser.add_argument('--correction_dropout', type=float, action='store', default=0.15,
                    help='Dropout for the top-2 correction MLP')
parser.add_argument('--correction_min_switch_precision', type=float, action='store', default=0.60,
                    help='Minimum dev precision required when selecting the SWITCH threshold for top-2 correction')
parser.add_argument('--correction_min_net_gain', type=float, action='store', default=1.0,
                    help='Minimum estimated dev net gain (#good_switch - #bad_switch) required to enable correction switching')
parser.add_argument('--correction_max_switch_rate', type=float, action='store', default=0.20,
                    help='Maximum fraction of dev samples allowed to be switched while calibrating the correction threshold')
parser.add_argument('--correction_threshold', type=float, action='store', default=-1.0,
                    help='Manual SWITCH probability threshold. If <0, threshold is calibrated on dev')
parser.add_argument('--correction_use_margin_gate', action='store_true',
                    help='If enabled, still require margin <= llm_replace_margin_threshold before applying correction SWITCH')
parser.add_argument('--multi_scale', action='store_true', help='Enable multi-scale graph distillation')
parser.add_argument('--re_init', action='store_true', help='Recompute cached graph node features')
args = parser.parse_args()
seed = args.seed

import numpy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

if seed > 0:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Set seed", seed)

import torch_geometric
from torch_scatter import scatter_mean
from graph_classes import *
from syntactic_graph import *
from semantic_graph import *
from MulCo_model_evidence_logic_llm_logaux_seed1103 import *
from MulCo_trainers_llm_logaux import BaseTrainer
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

syn_edge_type_dict = {'Doc-Sent': 0, 'Sent-Sent': 1, 'Sent-Word': 2, 'Word-Word': 3, 'Dependency': 4}
temporal_edge_type_dict = {'DCT-Timex': 0, 'Pred-Timex': 1, 'Timex-Timex': 2, 'Self-Loop': 3}


TEMPORAL_CUE_WORDS = {
    'before', 'after', 'when', 'while', 'during', 'until', 'since', 'once', 'then',
    'later', 'earlier', 'previously', 'following', 'followed', 'subsequently',
    'afterwards', 'meanwhile', 'prior', 'previous', 'subsequent', 'eventually',
    'already', 'soon', 'immediately', 'currently', 'recently', 'formerly'
}


def _normalize_for_cue(token):
    return re.sub(r'^[^A-Za-z]+|[^A-Za-z]+$', '', token.lower())


def get_temporal_cue_features(tokens, segment_indices):
    cue_features = []
    for start, end in segment_indices:
        seg_tokens = tokens[start:end]
        has_cue = any(_normalize_for_cue(token) in TEMPORAL_CUE_WORDS for token in seg_tokens)
        cue_features.append(1.0 if has_cue else 0.0)
    return cue_features

def build_llm_payload(tokens, e1_index, e2_index, doc_id, epair, window_size=12):
    """Create event-marked context and focused evidence snippets for LLM refinement."""
    def join_tokens(seq):
        return ' '.join(seq).strip()

    def mark_span(start, end):
        start = max(0, start)
        end = min(len(tokens), end)
        marked = []
        for idx in range(start, end):
            if idx == e1_index:
                marked.append('<EVENT1>')
            if idx == e2_index:
                marked.append('<EVENT2>')
            marked.append(tokens[idx])
            if idx == e1_index:
                marked.append('</EVENT1>')
            if idx == e2_index:
                marked.append('</EVENT2>')
        return join_tokens(marked)

    marked_tokens = []
    for idx, token in enumerate(tokens):
        if idx == e1_index:
            marked_tokens.append('<EVENT1>')
        if idx == e2_index:
            marked_tokens.append('<EVENT2>')
        marked_tokens.append(token)
        if idx == e1_index:
            marked_tokens.append('</EVENT1>')
        if idx == e2_index:
            marked_tokens.append('</EVENT2>')

    event1_text = tokens[e1_index] if 0 <= e1_index < len(tokens) else ''
    event2_text = tokens[e2_index] if 0 <= e2_index < len(tokens) else ''

    left_idx = min(e1_index, e2_index)
    right_idx = max(e1_index, e2_index)
    between_tokens = tokens[left_idx + 1:right_idx] if left_idx + 1 < right_idx else []
    cue_words = []
    for token in tokens:
        norm = _normalize_for_cue(token)
        if norm in TEMPORAL_CUE_WORDS and norm not in cue_words:
            cue_words.append(norm)

    return {
        'doc_id': doc_id,
        'epair': epair,
        'text': join_tokens(marked_tokens),
        'event1': event1_text,
        'event2': event2_text,
        'between_text': join_tokens(between_tokens) if between_tokens else '(no tokens between EVENT1 and EVENT2)',
        'event1_window': mark_span(e1_index - window_size, e1_index + window_size + 1),
        'event2_window': mark_span(e2_index - window_size, e2_index + window_size + 1),
        'focused_context': mark_span(max(0, left_idx - window_size), min(len(tokens), right_idx + window_size + 1)),
        'temporal_cues': cue_words,
    }


def _id_to_relation(rel_dict):
    return {v: k for k, v in rel_dict.items()}


def _label_choices_for_prompt(rel_dict):
    if 'i' in rel_dict or 'ii' in rel_dict or 's' in rel_dict:
        return [
            ('BEFORE', 'EVENT1 happens before EVENT2', 'b'),
            ('AFTER', 'EVENT1 happens after EVENT2', 'a'),
            ('SIMULTANEOUS', 'EVENT1 and EVENT2 happen at the same time', 's'),
            ('INCLUDES', 'EVENT1 temporally contains EVENT2', 'i'),
            ('IS_INCLUDED', 'EVENT1 is temporally contained in EVENT2', 'ii'),
            ('VAGUE', 'the relation is unclear or underspecified', 'v'),
        ]
    return [
        ('BEFORE', 'EVENT1 happens before EVENT2', 'b'),
        ('AFTER', 'EVENT1 happens after EVENT2', 'a'),
        ('EQUAL', 'EVENT1 and EVENT2 happen at the same time', 'e'),
        ('VAGUE', 'the relation is unclear or underspecified', 'v'),
    ]


def build_llm_prompt(payload, rel_dict):
    choices = _label_choices_for_prompt(rel_dict)
    choice_names = ', '.join([x[0] for x in choices if x[2] in rel_dict])
    definitions = '\n'.join([f'- {name}: {desc}.' for name, desc, key in choices if key in rel_dict])
    return (
        'You are a temporal relation classifier.\n'
        'Classify the temporal relation of EVENT1 to EVENT2 in the text.\n'
        f'Choose exactly one label from: {choice_names}.\n'
        f'{definitions}\n\n'
        f'Text: {payload["text"]}\n\n'
        'Answer with only one label.'
    )


def parse_llm_label(answer, rel_dict):
    text = answer.strip().lower()
    text = re.sub(r'[^a-z_\- ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Order matters: check contained forms before generic includes.
    if ('is included' in text or 'included in' in text or 'is_included' in text or 'ii' == text) and 'ii' in rel_dict:
        return rel_dict['ii']
    if ('simultaneous' in text or 'same time' in text or text == 's') and 's' in rel_dict:
        return rel_dict['s']
    if ('equal' in text or 'same time' in text or text == 'e') and 'e' in rel_dict:
        return rel_dict['e']
    if ('before' in text or text == 'b') and 'b' in rel_dict:
        return rel_dict['b']
    if ('after' in text or text == 'a') and 'a' in rel_dict:
        return rel_dict['a']
    if ('includes' in text or 'include' in text or text == 'i') and 'i' in rel_dict:
        return rel_dict['i']
    if ('vague' in text or 'unclear' in text or 'unknown' in text or text == 'v') and 'v' in rel_dict:
        return rel_dict['v']
    return None



def _relation_description(label_key):
    """Return a human-readable description for a relation key."""
    descriptions = {
        'a': 'EVENT1 happens AFTER EVENT2',
        'b': 'EVENT1 happens BEFORE EVENT2',
        'e': 'EVENT1 and EVENT2 happen at the same time',
        's': 'EVENT1 and EVENT2 happen at the same time',
        'i': 'EVENT1 temporally contains EVENT2',
        'ii': 'EVENT1 is temporally contained in EVENT2',
        'v': 'the relation is unclear or underspecified',
    }
    return descriptions.get(label_key, f'relation label {label_key}')


def _top2_from_record(record, rel_dict):
    """Get model top1/top2 label ids and probabilities from one confidence record."""
    probs = record.get('probs')
    if probs is None or len(probs) == 0:
        return record['pred'], None, float(record.get('confidence', 0.0)), 0.0

    ranked = sorted(range(len(probs)), key=lambda idx: probs[idx], reverse=True)
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None
    top1_prob = float(probs[top1])
    top2_prob = float(probs[top2]) if top2 is not None else 0.0
    return top1, top2, top1_prob, top2_prob


def build_llm_top2_judge_prompt(payload, rel_dict, option1_id, option2_id):
    """
    Build a blind top-2 judge prompt.

    Important:
    - The prompt does NOT tell the LLM which option is the neural model top-1.
    - The two candidate relations may be randomly swapped outside this function.
    - The LLM can only answer 1 or 2, so it cannot freely introduce vague or other labels.
    """
    id_to_rel = _id_to_relation(rel_dict)
    option1_key = id_to_rel.get(int(option1_id), str(option1_id))
    option2_key = id_to_rel.get(int(option2_id), str(option2_id))
    option1_desc = _relation_description(option1_key)
    option2_desc = _relation_description(option2_key)

    return (
        'You are a strict temporal relation judge.\n'
        'Your task is to judge the temporal relation of EVENT1 to EVENT2 in the text.\n'
        'Choose the more likely relation from ONLY the following two options.\n'
        'Do not use any other label. Do not explain.\n\n'
        f'Option 1: {option1_desc}.\n'
        f'Option 2: {option2_desc}.\n\n'
        f'Text: {payload["text"]}\n\n'
        'Answer with exactly one character: 1 or 2.\n'
        'Answer:'
    )



def _focused_evidence_block(payload):
    cues = payload.get('temporal_cues') or []
    cue_text = ', '.join(cues) if cues else 'none detected'
    return (
        f'Full marked context:\n{payload.get("text", "")}\n\n'
        f'Focused context around both events:\n{payload.get("focused_context", payload.get("text", ""))}\n\n'
        f'Text between EVENT1 and EVENT2:\n{payload.get("between_text", "")}\n\n'
        f'EVENT1 local context:\n{payload.get("event1_window", "")}\n\n'
        f'EVENT2 local context:\n{payload.get("event2_window", "")}\n\n'
        f'Detected temporal cue words: {cue_text}\n'
    )


def build_llm_context_top2_judge_prompt(payload, rel_dict, option1_id, option2_id):
    """Build a blind top-2 judge prompt with both full context and focused evidence snippets."""
    id_to_rel = _id_to_relation(rel_dict)
    option1_key = id_to_rel.get(int(option1_id), str(option1_id))
    option2_key = id_to_rel.get(int(option2_id), str(option2_id))
    option1_desc = _relation_description(option1_key)
    option2_desc = _relation_description(option2_key)
    evidence = _focused_evidence_block(payload)
    return (
        'You are a strict temporal relation judge for event temporal relation extraction.\n'
        'Use BOTH the full context and the focused evidence snippets.\n'
        'Choose the temporal relation of EVENT1 to EVENT2 from ONLY the two options.\n'
        'Do not explain. Do not use any other label.\n\n'
        f'{evidence}\n'
        f'Option 1: {option1_desc}.\n'
        f'Option 2: {option2_desc}.\n\n'
        'Answer with exactly one character: 1 or 2.\n'
        'Answer:'
    )


def build_llm_relation_verifier_prompt(payload, rel_dict, relation_id):
    """Ask the LLM whether a candidate relation is supported by the context/evidence."""
    id_to_rel = _id_to_relation(rel_dict)
    relation_key = id_to_rel.get(int(relation_id), str(relation_id))
    relation_desc = _relation_description(relation_key)
    evidence = _focused_evidence_block(payload)
    return (
        'You are a strict verifier for event temporal relation extraction.\n'
        'Decide whether the candidate temporal relation is supported by the full context and focused evidence.\n'
        'Answer only Correct or Incorrect. Do not explain.\n\n'
        f'{evidence}\n'
        f'Candidate relation: {relation_desc}.\n\n'
        'Answer with exactly one word: Correct or Incorrect.\n'
        'Answer:'
    )

def parse_llm_top2_choice(answer):
    """Parse a top-2 judge answer. Only accepts option 1 or option 2."""
    text = answer.strip().lower()
    text = re.sub(r'[^a-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Prefer explicit numeric answers.
    if re.fullmatch(r'(option )?1', text) or text.startswith('1 '):
        return 1
    if re.fullmatch(r'(option )?2', text) or text.startswith('2 '):
        return 2

    # Fallback for short natural outputs.
    if 'option 1' in text or 'choice 1' in text:
        return 1
    if 'option 2' in text or 'choice 2' in text:
        return 2
    return None



def _answer_logprob_causal(llm_model, llm_tokenizer, prompt, answer_text, device, max_input_length):
    """Score an answer string by causal-LM log probability conditioned on the prompt."""
    prompt_encoded = llm_tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=max_input_length
    ).to(device)
    answer_ids = llm_tokenizer(answer_text, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)
    if answer_ids.numel() == 0:
        return float('-inf')

    input_ids = torch.cat([prompt_encoded['input_ids'], answer_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, device=device)
    outputs = llm_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    prompt_len = prompt_encoded['input_ids'].shape[1]
    answer_len = answer_ids.shape[1]
    log_probs = F.log_softmax(logits[:, prompt_len - 1: prompt_len + answer_len - 1, :], dim=-1)
    token_log_probs = log_probs.gather(2, answer_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.mean().detach().cpu())


def _answer_logprob_seq2seq(llm_model, llm_tokenizer, prompt, answer_text, device, max_input_length):
    """Score an answer string by seq2seq negative loss."""
    encoded = llm_tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=max_input_length
    ).to(device)
    labels = llm_tokenizer(answer_text, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)
    if labels.numel() == 0:
        return float('-inf')
    outputs = llm_model(**encoded, labels=labels)
    return float((-outputs.loss).detach().cpu())


def score_top2_choice(llm_model, llm_tokenizer, prompt, device, llm_is_encoder_decoder, max_input_length):
    """
    Token-level top2 judge.
    It never asks the LLM to generate explanations. It compares the conditional scores of answer 1 and answer 2.
    """
    # Different chat/causal tokenizers may prefer either bare or space-prefixed numbers.
    answer_variants = {
        1: ['1', ' 1'],
        2: ['2', ' 2'],
    }
    scores = {}
    for choice, variants in answer_variants.items():
        variant_scores = []
        for ans in variants:
            try:
                if llm_is_encoder_decoder:
                    variant_scores.append(_answer_logprob_seq2seq(llm_model, llm_tokenizer, prompt, ans, device, max_input_length))
                else:
                    variant_scores.append(_answer_logprob_causal(llm_model, llm_tokenizer, prompt, ans, device, max_input_length))
            except Exception:
                variant_scores.append(float('-inf'))
        scores[choice] = max(variant_scores)

    if scores[1] == float('-inf') and scores[2] == float('-inf'):
        return None, scores
    return (1 if scores[1] >= scores[2] else 2), scores



def score_top2_double_order_choice(llm_model, llm_tokenizer, payload, rel_dict, top1_id, top2_id,
                                   device, llm_is_encoder_decoder, max_input_length,
                                   use_context_evidence=False):
    """Score top1/top2 in both option orders to reduce Option-1/Option-2 position bias.

    Order A: Option 1 = top1, Option 2 = top2
    Order B: Option 1 = top2, Option 2 = top1

    The final top1/top2 score is the average score of the same relation across both orders.
    Returned choice uses the normalized order: 1 means top1, 2 means top2.
    """
    prompt_builder = build_llm_context_top2_judge_prompt if use_context_evidence else build_llm_top2_judge_prompt

    prompt_a = prompt_builder(payload, rel_dict, top1_id, top2_id)
    choice_a, scores_a = score_top2_choice(
        llm_model, llm_tokenizer, prompt_a, device, llm_is_encoder_decoder, max_input_length
    )

    prompt_b = prompt_builder(payload, rel_dict, top2_id, top1_id)
    choice_b, scores_b = score_top2_choice(
        llm_model, llm_tokenizer, prompt_b, device, llm_is_encoder_decoder, max_input_length
    )

    if scores_a.get(1, float('-inf')) == float('-inf') and scores_a.get(2, float('-inf')) == float('-inf'):
        return None, {1: float('-inf'), 2: float('-inf')}, {'order_a_failed': True, 'order_b_failed': False}
    if scores_b.get(1, float('-inf')) == float('-inf') and scores_b.get(2, float('-inf')) == float('-inf'):
        return None, {1: float('-inf'), 2: float('-inf')}, {'order_a_failed': False, 'order_b_failed': True}

    # Normalize to relation-level scores.
    # top1 is option 1 in order A and option 2 in order B.
    # top2 is option 2 in order A and option 1 in order B.
    top1_score = (scores_a[1] + scores_b[2]) / 2.0
    top2_score = (scores_a[2] + scores_b[1]) / 2.0
    normalized_scores = {1: top1_score, 2: top2_score}
    normalized_choice = 1 if top1_score >= top2_score else 2

    details = {
        'order_a_choice': choice_a,
        'order_a_score_1': scores_a.get(1),
        'order_a_score_2': scores_a.get(2),
        'order_b_choice': choice_b,
        'order_b_score_1': scores_b.get(1),
        'order_b_score_2': scores_b.get(2),
        'double_top1_score': top1_score,
        'double_top2_score': top2_score,
    }
    return normalized_choice, normalized_scores, details



def score_binary_verdict(llm_model, llm_tokenizer, prompt, device, llm_is_encoder_decoder, max_input_length):
    """Return support score = logprob(Correct) - logprob(Incorrect)."""
    correct_variants = ['Correct', ' Correct', 'correct', ' correct']
    incorrect_variants = ['Incorrect', ' Incorrect', 'incorrect', ' incorrect']

    def best_score(variants):
        scores = []
        for ans in variants:
            try:
                if llm_is_encoder_decoder:
                    scores.append(_answer_logprob_seq2seq(llm_model, llm_tokenizer, prompt, ans, device, max_input_length))
                else:
                    scores.append(_answer_logprob_causal(llm_model, llm_tokenizer, prompt, ans, device, max_input_length))
            except Exception:
                scores.append(float('-inf'))
        return max(scores)

    correct_score = best_score(correct_variants)
    incorrect_score = best_score(incorrect_variants)
    if correct_score == float('-inf') and incorrect_score == float('-inf'):
        return None, {'Correct': correct_score, 'Incorrect': incorrect_score}
    return correct_score - incorrect_score, {'Correct': correct_score, 'Incorrect': incorrect_score}




def _branch_summary(branch, id_to_rel):
    if not branch:
        return 'not available'
    top1 = branch.get('top1')
    top2 = branch.get('top2')
    top1_label = id_to_rel.get(int(top1), str(top1)) if top1 is not None else 'NA'
    top2_label = id_to_rel.get(int(top2), str(top2)) if top2 is not None else 'NA'
    return f"top1={top1_label}({float(branch.get('top1_prob', 0.0)):.3f}), top2={top2_label}({float(branch.get('top2_prob', 0.0)):.3f})"


def _branch_disagreement(record, top1_id, top2_id):
    """Return True if BERT/GNN branch predictions disagree with final top1 or support top2."""
    disagreement = False
    support_top2 = False
    bert = record.get('bert_branch')
    gcn = record.get('gcn_branch')
    for branch in [bert, gcn]:
        if not branch:
            continue
        b_top1 = branch.get('top1')
        if b_top1 is None:
            continue
        if int(b_top1) != int(top1_id):
            disagreement = True
        if top2_id is not None and int(b_top1) == int(top2_id):
            support_top2 = True
    if bert and gcn and bert.get('top1') is not None and gcn.get('top1') is not None:
        if int(bert.get('top1')) != int(gcn.get('top1')):
            disagreement = True
    return disagreement or support_top2


def build_llm_log_guided_ksu_prompt(payload, rel_dict, record, top1_id, top2_id):
    """Decision-log guided KEEP/SWITCH/UNCERTAIN prompt."""
    id_to_rel = _id_to_relation(rel_dict)
    top1_key = id_to_rel.get(int(top1_id), str(top1_id))
    top2_key = id_to_rel.get(int(top2_id), str(top2_id))
    top1_desc = _relation_description(top1_key)
    top2_desc = _relation_description(top2_key)
    evidence = _focused_evidence_block(payload)
    final_probs = record.get('probs') or []
    top2_prob = float(final_probs[int(top2_id)]) if final_probs and int(top2_id) < len(final_probs) else 0.0
    bert_summary = _branch_summary(record.get('bert_branch'), id_to_rel)
    gcn_summary = _branch_summary(record.get('gcn_branch'), id_to_rel)
    disagreement = _branch_disagreement(record, top1_id, top2_id)
    return (
        'You are auditing a temporal relation prediction.\n'
        'Use the context, focused evidence, and model decision log.\n'
        'Your job is NOT to freely classify all labels. Choose only one action.\n\n'
        f'{evidence}\n'
        'Model decision log:\n'
        f'- Main prediction top1: {top1_desc}, probability={float(record.get("confidence", 0.0)):.3f}.\n'
        f'- Main prediction top2: {top2_desc}, probability={top2_prob:.3f}.\n'
        f'- Main top1-top2 margin: {float(record.get("margin", 0.0)):.3f}.\n'
        f'- BERT branch: {bert_summary}.\n'
        f'- GNN branch: {gcn_summary}.\n'
        f'- Branch disagreement or branch top2 support: {str(bool(disagreement))}.\n\n'
        'Actions:\n'
        'K = KEEP the main prediction top1.\n'
        'S = SWITCH to the second candidate top2.\n'
        'U = UNCERTAIN; keep the main prediction.\n\n'
        'Choose S only if the evidence and branch log clearly support top2 over top1.\n'
        'If evidence is ambiguous, choose U.\n'
        'Answer with exactly one character: K, S, or U.\n'
        'Answer:'
    )


def score_ksu_choice(llm_model, llm_tokenizer, prompt, device, llm_is_encoder_decoder, max_input_length):
    """Score KEEP/SWITCH/UNCERTAIN by conditional token log probability."""
    answer_variants = {
        'K': ['K', ' K', 'KEEP', ' KEEP'],
        'S': ['S', ' S', 'SWITCH', ' SWITCH'],
        'U': ['U', ' U', 'UNCERTAIN', ' UNCERTAIN'],
    }
    scores = {}
    for choice, variants in answer_variants.items():
        variant_scores = []
        for ans in variants:
            try:
                if llm_is_encoder_decoder:
                    variant_scores.append(_answer_logprob_seq2seq(llm_model, llm_tokenizer, prompt, ans, device, max_input_length))
                else:
                    variant_scores.append(_answer_logprob_causal(llm_model, llm_tokenizer, prompt, ans, device, max_input_length))
            except Exception:
                variant_scores.append(float('-inf'))
        scores[choice] = max(variant_scores)
    best = max(scores, key=scores.get)
    return best, scores

def _collect_labels_from_batches(batches):
    """Flatten gold labels from batched examples in the same order used by test_with_confidence."""
    labels = []
    for batch in batches:
        if 'labels' in batch[0]:
            labels.extend(batch[0]['labels'].detach().cpu().numpy().tolist())
    return labels



# Global dev-calibrated risk profile used by risk-ranked LLM candidate selector.
LLM_RISK_PROFILE = {}


def _parse_float_list(value, default=None):
    if default is None:
        default = [0.05, 0.08, 0.12, 0.20]
    try:
        xs = [float(x.strip()) for x in str(value).split(',') if x.strip() != '']
        xs = sorted(set(xs))
        return xs if xs else default
    except Exception:
        return default


def _margin_bin_label(margin, bins):
    margin = float(margin)
    prev = 0.0
    for b in bins:
        b = float(b)
        if margin <= b:
            return f'({prev:.4f},{b:.4f}]'
        prev = b
    return f'({prev:.4f},inf)'


def _safe_entropy_norm(probs):
    if probs is None or len(probs) == 0:
        return 0.0
    ps = numpy.asarray(probs, dtype=float)
    ps = numpy.clip(ps, 1e-12, 1.0)
    ps = ps / ps.sum()
    ent = float(-(ps * numpy.log(ps)).sum())
    denom = float(numpy.log(len(ps))) if len(ps) > 1 else 1.0
    return max(0.0, min(1.0, ent / denom))


def _risk_profile_keys(record, rel_dict, args):
    id_to_rel = _id_to_relation(rel_dict)
    top1_id, top2_id, _, _ = _top2_from_record(record, rel_dict)
    top1_key = id_to_rel.get(int(top1_id), str(top1_id)) if top1_id is not None else 'NA'
    top2_key = id_to_rel.get(int(top2_id), str(top2_id)) if top2_id is not None else 'NA'
    bins = _parse_float_list(getattr(args, 'llm_dev_error_profile_bins', '0.05,0.08,0.12,0.20'))
    margin_bin = _margin_bin_label(float(record.get('margin', 1.0)), bins)
    pair_key = f'top1={top1_key}|top2={top2_key}'
    profile_key = f'{pair_key}|margin={margin_bin}'
    return profile_key, pair_key, margin_bin


def _compute_profile_error_rate(counts, key, fallback=0.0):
    """Return a smoothed dev error rate for a profile key.

    `fallback` can be None when a fine-grained key is missing during
    calibration. In that case we fall back to 0.0 here and let the caller
    optionally replace the result with a broader pair-level rate.
    """
    if fallback is None:
        fallback = 0.0
    item = counts.get(key) if counts is not None else None
    if not item:
        return float(fallback)
    total = int(item.get('total', 0))
    wrong = int(item.get('wrong', 0))
    if total <= 0:
        return float(fallback)
    global_rate = float(fallback)
    return float((wrong + global_rate) / (total + 1.0))


def _llm_error_risk_score(record, rel_dict, args):
    margin = float(record.get('margin', 1.0))
    conf = float(record.get('confidence', 1.0))
    probs = record.get('probs') or []
    margin_cap = max(1e-6, float(getattr(args, 'llm_risk_margin_cap', 0.20)))
    margin_risk = max(0.0, min(1.0, 1.0 - margin / margin_cap))
    conf_risk = max(0.0, min(1.0, 1.0 - conf))
    entropy_risk = _safe_entropy_norm(probs)

    profile_key, pair_key, margin_bin = _risk_profile_keys(record, rel_dict, args)
    profile_data = LLM_RISK_PROFILE or {}
    global_error_rate = float(profile_data.get('global_error_rate', 0.0))
    profile_counts = profile_data.get('profile_counts', {})
    pair_counts = profile_data.get('pair_counts', {})
    profile_rate = _compute_profile_error_rate(profile_counts, profile_key, global_error_rate) if profile_counts else global_error_rate
    pair_rate = _compute_profile_error_rate(pair_counts, pair_key, global_error_rate) if pair_counts else global_error_rate
    profile_risk = max(0.0, min(1.0, float(profile_rate)))

    weights = {
        'margin': float(getattr(args, 'llm_margin_weight', 0.35)),
        'confidence': float(getattr(args, 'llm_conf_weight', 0.20)),
        'entropy': float(getattr(args, 'llm_entropy_weight', 0.25)),
        'profile': float(getattr(args, 'llm_profile_weight', 0.20)),
    }
    total_w = sum(max(0.0, w) for w in weights.values()) or 1.0
    score = (
        max(0.0, weights['margin']) * margin_risk +
        max(0.0, weights['confidence']) * conf_risk +
        max(0.0, weights['entropy']) * entropy_risk +
        max(0.0, weights['profile']) * profile_risk
    ) / total_w
    score = max(0.0, min(1.0, float(score)))
    return score, {
        'risk_score': score,
        'margin_risk': margin_risk,
        'confidence_risk': conf_risk,
        'entropy_risk': entropy_risk,
        'profile_risk': profile_risk,
        'profile_key': profile_key,
        'pair_key': pair_key,
        'margin_bin': margin_bin,
        'global_error_rate': global_error_rate,
    }




# Global dev-trained deferral network profile. This is intentionally small and trained
# only from dev correctness signals, so test gold labels are never used.
LLM_DEFER_PROFILE = {}


class _DeferralMLP(nn.Module):
    def __init__(self, input_dim, hidden_size=32, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _defer_feature_vector(record, rel_dict, args):
    """Feature vector for the deferral network.

    The deferral MLP should learn *where the base model is likely wrong*, not
    how to classify the temporal relation. Therefore we use uncertainty/risk
    features and top1/top2 identities, but we do not use gold labels at test time.
    """
    num_classes = len(rel_dict)
    probs = record.get('probs') or []
    probs = list(probs)[:num_classes]
    if len(probs) < num_classes:
        probs = probs + [0.0] * (num_classes - len(probs))
    ps = numpy.asarray(probs, dtype=float)
    if ps.sum() <= 0:
        ps = numpy.ones(num_classes, dtype=float) / max(1, num_classes)
    else:
        ps = ps / ps.sum()

    top1_id, top2_id, top1_prob, top2_prob = _top2_from_record(record, rel_dict)
    top1_id = int(top1_id) if top1_id is not None else int(numpy.argmax(ps))
    top2_id = int(top2_id) if top2_id is not None else int(numpy.argsort(ps)[-2]) if num_classes > 1 else top1_id
    top1_oh = numpy.zeros(num_classes, dtype=float)
    top2_oh = numpy.zeros(num_classes, dtype=float)
    if 0 <= top1_id < num_classes:
        top1_oh[top1_id] = 1.0
    if 0 <= top2_id < num_classes:
        top2_oh[top2_id] = 1.0

    risk_score, risk_info = _llm_error_risk_score(record, rel_dict, args)
    confidence = float(record.get('confidence', top1_prob if top1_prob is not None else ps.max()))
    margin = float(record.get('margin', abs(float(top1_prob or 0.0) - float(top2_prob or 0.0))))
    entropy = _safe_entropy_norm(ps)
    base_feats = numpy.asarray([
        confidence,
        margin,
        1.0 - confidence,
        entropy,
        risk_score,
        float(risk_info.get('margin_risk', 0.0)),
        float(risk_info.get('confidence_risk', 0.0)),
        float(risk_info.get('entropy_risk', 0.0)),
        float(risk_info.get('profile_risk', 0.0)),
        float(top1_prob if top1_prob is not None else ps[top1_id]),
        float(top2_prob if top2_prob is not None else ps[top2_id]),
        float((top1_prob if top1_prob is not None else ps[top1_id]) - (top2_prob if top2_prob is not None else ps[top2_id])),
    ], dtype=float)
    return numpy.concatenate([base_feats, ps, top1_oh, top2_oh], axis=0)


def train_llm_deferral_mlp_from_dev(dev_confidence_records, dev_labels, rel_dict, args, model_path):
    """Train a small MLP to predict whether the base model should defer to LLM.

    Positive label on dev = base prediction is wrong. This does not use test labels.
    The learned network is used only to select test candidates for LLM K/S/U audit.
    """
    global LLM_DEFER_PROFILE
    if dev_confidence_records is None or dev_labels is None or len(dev_confidence_records) == 0:
        LLM_DEFER_PROFILE = {'enabled': False, 'reason': 'empty_dev_records'}
        return LLM_DEFER_PROFILE

    # Build risk profile first if it has not been calibrated. The deferral feature
    # vector uses profile_risk as a soft feature.
    global LLM_RISK_PROFILE
    if not LLM_RISK_PROFILE:
        paired_tmp = []
        profile_counts, pair_counts = {}, {}
        for rec, gold in zip(dev_confidence_records, dev_labels):
            pred = int(rec.get('pred'))
            gold = int(gold)
            correct = pred == gold
            profile_key, pair_key, _ = _risk_profile_keys(rec, rel_dict, args)
            for counts, key in [(profile_counts, profile_key), (pair_counts, pair_key)]:
                if key not in counts:
                    counts[key] = {'total': 0, 'wrong': 0}
                counts[key]['total'] += 1
                counts[key]['wrong'] += 0 if correct else 1
            paired_tmp.append(correct)
        global_error_rate = (sum(1 for c in paired_tmp if not c) / len(paired_tmp)) if paired_tmp else 0.0
        LLM_RISK_PROFILE = {
            'global_error_rate': global_error_rate,
            'profile_counts': profile_counts,
            'pair_counts': pair_counts,
            'target_rate': None,
            'target_count_on_dev': None,
        }

    X, y, meta = [], [], []
    for rec, gold in zip(dev_confidence_records, dev_labels):
        try:
            feat = _defer_feature_vector(rec, rel_dict, args)
        except Exception:
            continue
        pred = int(rec.get('pred'))
        gold = int(gold)
        wrong = 1 if pred != gold else 0
        X.append(feat)
        y.append(wrong)
        meta.append({
            'doc_id': rec.get('doc_id'),
            'epair': rec.get('epair'),
            'pred': pred,
            'gold': gold,
            'wrong': bool(wrong),
            'margin': float(rec.get('margin', 1.0)),
            'confidence': float(rec.get('confidence', 1.0)),
        })
    if not X:
        LLM_DEFER_PROFILE = {'enabled': False, 'reason': 'empty_features'}
        return LLM_DEFER_PROFILE

    X = numpy.vstack(X).astype('float32')
    y = numpy.asarray(y, dtype='float32')
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std

    device = torch.device('cpu')
    model = _DeferralMLP(Xn.shape[1], int(getattr(args, 'llm_defer_hidden_size', 32)), float(getattr(args, 'llm_defer_dropout', 0.15))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(getattr(args, 'llm_defer_lr', 1e-3)), weight_decay=1e-4)
    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    model.train()
    for _ in range(int(getattr(args, 'llm_defer_epochs', 250))):
        optimizer.zero_grad()
        logits = model(xt)
        loss = criterion(logits, yt)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(xt)).cpu().numpy()
    ranked_idx = list(numpy.argsort(-probs))
    n = len(ranked_idx)
    max_by_rate = int(numpy.ceil(float(getattr(args, 'llm_defer_max_candidate_rate', 0.20)) * n))
    max_by_arg = int(getattr(args, 'llm_candidate_max_count', 30)) if int(getattr(args, 'llm_candidate_max_count', 30)) > 0 else n
    max_k = min(n, max(max_by_rate, int(getattr(args, 'llm_defer_min_candidates', 3))), max_by_arg)
    min_prec = float(getattr(args, 'llm_defer_min_error_precision', 0.40))
    chosen_k, chosen_precision, chosen_recall = 0, 0.0, 0.0
    prefix_stats = []
    total_wrong = int(y.sum())
    for k in range(1, max_k + 1):
        ids = ranked_idx[:k]
        wrong = int(y[ids].sum())
        precision = wrong / k if k else 0.0
        recall = wrong / max(1, total_wrong)
        prefix_stats.append({'k': k, 'precision': precision, 'error_recall': recall})
        if precision >= min_prec:
            chosen_k, chosen_precision, chosen_recall = k, precision, recall
    if chosen_k == 0:
        # Fallback: choose the k with best precision, then larger recall.
        best = max(prefix_stats, key=lambda z: (z['precision'], z['error_recall'])) if prefix_stats else {'k': min(max_k, n), 'precision': 0.0, 'error_recall': 0.0}
        chosen_k = int(best['k'])
        chosen_precision = float(best['precision'])
        chosen_recall = float(best['error_recall'])

    preview = []
    for idx in ranked_idx[:min(50, n)]:
        item = dict(meta[idx])
        item['defer_prob'] = float(probs[idx])
        preview.append(item)

    LLM_DEFER_PROFILE = {
        'enabled': True,
        'model': model,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'input_dim': int(Xn.shape[1]),
        'num_dev_samples': int(n),
        'num_dev_errors': int(total_wrong),
        'dev_error_rate': float(total_wrong / max(1, n)),
        'target_count_on_dev': int(chosen_k),
        'target_rate': float(chosen_k / max(1, n)),
        'target_error_precision_on_dev': float(chosen_precision),
        'target_error_recall_on_dev': float(chosen_recall),
        'min_error_precision': float(min_prec),
        'prefix_stats': prefix_stats[:50],
        'preview': preview,
    }
    torch.save({'state_dict': model.state_dict(), 'mean': mean, 'std': std, 'input_dim': Xn.shape[1]}, os.path.join(model_path, 'llm_deferral_mlp.pt'))
    json.dump({k: v for k, v in LLM_DEFER_PROFILE.items() if k != 'model'}, open(os.path.join(model_path, 'llm_deferral_mlp_stats.json'), 'w'), indent=4)
    print('Dev-trained deferral MLP:')
    print(f"  dev samples/errors: {n}/{total_wrong}")
    print(f"  target_count: {chosen_k} ({chosen_k / max(1, n):.2%})")
    print(f"  dev candidate error precision: {chosen_precision:.2%}")
    print(f"  dev error recall among defer candidates: {chosen_recall:.2%}")
    return LLM_DEFER_PROFILE


def predict_defer_probability(record, rel_dict, args):
    profile = LLM_DEFER_PROFILE or {}
    model = profile.get('model')
    if model is None:
        return 0.0, {'defer_reason': 'not_trained'}
    feat = _defer_feature_vector(record, rel_dict, args).astype('float32')
    mean = numpy.asarray(profile.get('mean'), dtype='float32')
    std = numpy.asarray(profile.get('std'), dtype='float32')
    if mean.shape[0] != feat.shape[0]:
        return 0.0, {'defer_reason': 'feature_dim_mismatch'}
    x = (feat - mean) / numpy.where(std < 1e-6, 1.0, std)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(x[None, :], dtype=torch.float32))).item()
    risk_score, risk_info = _llm_error_risk_score(record, rel_dict, args)
    return float(prob), {'defer_prob': float(prob), 'risk_score': float(risk_score), 'risk_info': risk_info}



# Global dev-trained top-2 correction network.
# Unlike the deferral MLP, this model predicts whether changing base top1 to base top2 is likely beneficial.
TOP2_CORRECTION_PROFILE = {}


class _Top2CorrectionMLP(nn.Module):
    def __init__(self, input_dim, hidden_size=32, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _correction_feature_vector(record, rel_dict, args):
    """Feature vector for the top-2 correction network.

    It uses model uncertainty/risk features plus the learned defer probability.
    It never uses gold labels at test time.
    """
    base = _defer_feature_vector(record, rel_dict, args).astype('float32')
    # If the deferral MLP is already trained, its probability is a useful hard-sample feature.
    try:
        defer_prob, defer_info = predict_defer_probability(record, rel_dict, args)
    except Exception:
        defer_prob, defer_info = 0.0, {}
    risk_score, risk_info = _llm_error_risk_score(record, rel_dict, args)
    extra = numpy.asarray([
        float(defer_prob),
        float(risk_score),
        float(risk_info.get('profile_risk', 0.0)),
        float(risk_info.get('margin_risk', 0.0)),
    ], dtype='float32')
    return numpy.concatenate([base, extra], axis=0)


def _correction_target(record, gold, rel_dict):
    """Return 1 only when switching from top1 to top2 would fix the sample.

    Positive: top1 is wrong AND top2 equals gold.
    Negative: top1 is correct OR top2 is also wrong.
    """
    top1_id, top2_id, _, _ = _top2_from_record(record, rel_dict)
    if top2_id is None:
        return None
    gold = int(gold)
    top1_id = int(top1_id)
    top2_id = int(top2_id)
    return 1 if (top1_id != gold and top2_id == gold) else 0


def train_top2_correction_mlp_from_dev(dev_confidence_records, dev_labels, rel_dict, args, model_path):
    """Train a dev-supervised top-2 correction MLP.

    This model learns whether to KEEP the base top1 prediction or SWITCH to top2.
    It uses dev labels only for calibration/training and never uses test labels.
    """
    global TOP2_CORRECTION_PROFILE
    if dev_confidence_records is None or dev_labels is None or len(dev_confidence_records) == 0:
        TOP2_CORRECTION_PROFILE = {'enabled': False, 'reason': 'empty_dev_records'}
        return TOP2_CORRECTION_PROFILE

    X, y, meta = [], [], []
    for rec, gold in zip(dev_confidence_records, dev_labels):
        target = _correction_target(rec, gold, rel_dict)
        if target is None:
            continue
        try:
            feat = _correction_feature_vector(rec, rel_dict, args)
        except Exception:
            continue
        top1_id, top2_id, top1_prob, top2_prob = _top2_from_record(rec, rel_dict)
        pred = int(top1_id)
        gold_i = int(gold)
        X.append(feat)
        y.append(float(target))
        meta.append({
            'doc_id': rec.get('doc_id'),
            'epair': rec.get('epair'),
            'pred': pred,
            'top2': int(top2_id),
            'gold': gold_i,
            'switch_good': bool(target),
            'top1_correct': bool(pred == gold_i),
            'top2_correct': bool(int(top2_id) == gold_i),
            'margin': float(rec.get('margin', 1.0)),
            'confidence': float(rec.get('confidence', 1.0)),
        })

    if not X:
        TOP2_CORRECTION_PROFILE = {'enabled': False, 'reason': 'empty_features'}
        return TOP2_CORRECTION_PROFILE

    X = numpy.vstack(X).astype('float32')
    y = numpy.asarray(y, dtype='float32')
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std

    device = torch.device('cpu')
    model = _Top2CorrectionMLP(
        Xn.shape[1],
        int(getattr(args, 'correction_hidden_size', 32)),
        float(getattr(args, 'correction_dropout', 0.15))
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(getattr(args, 'correction_lr', 1e-3)), weight_decay=1e-4)
    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    model.train()
    for _ in range(int(getattr(args, 'correction_epochs', 300))):
        optimizer.zero_grad()
        logits = model(xt)
        loss = criterion(logits, yt)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(xt)).cpu().numpy()

    # Calibrate switch threshold on dev. A switch is good if top2==gold and top1!=gold.
    # It is harmful if top1 was originally correct. If both top1/top2 are wrong, switch is neutral.
    manual_thr = float(getattr(args, 'correction_threshold', -1.0))
    max_switch_rate = float(getattr(args, 'correction_max_switch_rate', 0.20))
    min_prec = float(getattr(args, 'correction_min_switch_precision', 0.60))
    min_gain = float(getattr(args, 'correction_min_net_gain', 1.0))
    n = len(y)
    top1_correct_arr = numpy.asarray([1 if m['top1_correct'] else 0 for m in meta], dtype=int)
    top2_good_arr = y.astype(int)
    thresholds = sorted(set([float(x) for x in probs.tolist()] + [0.5, 0.6, 0.7, 0.8, 0.9]), reverse=True)
    calib_rows = []
    best = None
    for thr in thresholds:
        sel = probs >= thr
        count = int(sel.sum())
        if count == 0:
            continue
        if count > max(1, int(numpy.ceil(max_switch_rate * n))):
            continue
        good = int(top2_good_arr[sel].sum())
        bad = int(top1_correct_arr[sel].sum())
        neutral = int(count - good - bad)
        precision = good / max(1, count)
        net_gain = good - bad
        row = {
            'threshold': float(thr),
            'count': count,
            'good_switch': good,
            'bad_switch': bad,
            'neutral_switch': neutral,
            'precision': float(precision),
            'net_gain': int(net_gain),
        }
        calib_rows.append(row)
        if precision >= min_prec and net_gain >= min_gain:
            key = (net_gain, precision, -count, thr)
            if best is None or key > best[0]:
                best = (key, row)

    if manual_thr >= 0:
        chosen_threshold = manual_thr
        threshold_reason = 'manual'
    elif best is not None:
        chosen_threshold = float(best[1]['threshold'])
        threshold_reason = 'dev_positive_gain'
    else:
        # No dev setting produced positive estimated gain; disable switching by default.
        chosen_threshold = 1.01
        threshold_reason = 'disabled_no_positive_dev_gain'

    ranked_idx = list(numpy.argsort(-probs))
    preview = []
    for idx in ranked_idx[:min(50, n)]:
        item = dict(meta[idx])
        item['correction_switch_prob'] = float(probs[idx])
        preview.append(item)

    TOP2_CORRECTION_PROFILE = {
        'enabled': True,
        'model': model,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'input_dim': int(Xn.shape[1]),
        'num_dev_samples': int(n),
        'num_dev_switch_good': int(y.sum()),
        'switch_good_rate': float(y.mean()) if len(y) else 0.0,
        'chosen_threshold': float(chosen_threshold),
        'threshold_reason': threshold_reason,
        'min_switch_precision': float(min_prec),
        'min_net_gain': float(min_gain),
        'max_switch_rate': float(max_switch_rate),
        'calibration_rows': calib_rows[:100],
        'preview': preview,
    }
    torch.save({'state_dict': model.state_dict(), 'mean': mean, 'std': std, 'input_dim': Xn.shape[1]}, os.path.join(model_path, 'top2_correction_mlp.pt'))
    json.dump({k: v for k, v in TOP2_CORRECTION_PROFILE.items() if k != 'model'}, open(os.path.join(model_path, 'top2_correction_mlp_stats.json'), 'w'), indent=4)
    print('Dev-trained top-2 correction MLP:')
    print(f"  dev samples / switch-good samples: {n}/{int(y.sum())}")
    print(f"  chosen switch threshold: {chosen_threshold:.4f} ({threshold_reason})")
    return TOP2_CORRECTION_PROFILE


def predict_top2_correction_probability(record, rel_dict, args):
    profile = TOP2_CORRECTION_PROFILE or {}
    model = profile.get('model')
    if model is None:
        return 0.0, {'correction_reason': 'not_trained'}
    feat = _correction_feature_vector(record, rel_dict, args).astype('float32')
    mean = numpy.asarray(profile.get('mean'), dtype='float32')
    std = numpy.asarray(profile.get('std'), dtype='float32')
    if mean.shape[0] != feat.shape[0]:
        return 0.0, {'correction_reason': 'feature_dim_mismatch'}
    x = (feat - mean) / numpy.where(std < 1e-6, 1.0, std)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(x[None, :], dtype=torch.float32))).item()
    return float(prob), {'correction_switch_prob': float(prob)}


def apply_top2_correction_refinement(predictions, candidates, rel_dict, args, model_path, stats):
    """Apply dev-trained Top-2 Correction Network instead of LLM.

    It only decides KEEP vs SWITCH(top2) for candidates selected by deferral/risk logic.
    """
    id_to_rel = _id_to_relation(rel_dict)
    threshold = float((TOP2_CORRECTION_PROFILE or {}).get('chosen_threshold', getattr(args, 'correction_threshold', 1.01)))
    stats['correction_mlp_enabled'] = True
    stats['correction_threshold'] = threshold
    stats['correction_threshold_reason'] = (TOP2_CORRECTION_PROFILE or {}).get('threshold_reason')
    stats['correction_switch_count'] = 0
    stats['correction_keep_count'] = 0
    stats['correction_margin_gate'] = bool(getattr(args, 'correction_use_margin_gate', False))
    for record in tqdm(candidates, desc='Top-2 correction reselecting candidates'):
        payload = record.get('payload') or {}
        base_pred = int(record.get('pred'))
        top1_id, top2_id, top1_prob, top2_prob = _top2_from_record(record, rel_dict)
        top1_id = int(top1_id)
        top2_id = int(top2_id) if top2_id is not None else None
        final_pred = base_pred
        decision_reason = 'not_processed'
        switch_prob, switch_info = predict_top2_correction_probability(record, rel_dict, args)
        if top2_id is None:
            stats['failed_count'] += 1
            decision_reason = 'no_top2'
        else:
            proposed_key = id_to_rel.get(top2_id, str(top2_id))
            if switch_prob < threshold:
                stats['blocked_count'] += 1
                stats['correction_keep_count'] += 1
                decision_reason = 'correction_keep_top1'
            elif proposed_key == 'v' and not args.allow_llm_vague_replace:
                stats['blocked_count'] += 1
                decision_reason = 'blocked_vague_replacement'
            elif getattr(args, 'correction_use_margin_gate', False) and record['margin'] > args.llm_replace_margin_threshold:
                stats['blocked_count'] += 1
                decision_reason = 'blocked_margin_too_large'
            else:
                final_pred = top2_id
                stats['correction_switch_count'] += 1
                stats['top2_choice_count'] += 1
                decision_reason = 'replaced_by_top2_correction_mlp'
        if final_pred != base_pred:
            predictions[record['doc_id']][record['epair']] = final_pred
            stats['changed_count'] += 1
        stats['records'].append({
            'doc_id': record.get('doc_id'),
            'epair': list(record.get('epair')),
            'confidence': record.get('confidence'),
            'margin': record.get('margin'),
            'top1_prob': top1_prob,
            'top2_prob': top2_prob,
            'top1_label': id_to_rel.get(top1_id, str(top1_id)),
            'top2_label': id_to_rel.get(top2_id, str(top2_id)) if top2_id is not None else None,
            'base_pred': id_to_rel.get(base_pred, str(base_pred)),
            'final_pred': id_to_rel.get(final_pred, str(final_pred)),
            'decision': decision_reason,
            'correction_switch_prob': float(switch_prob),
            'correction_info': switch_info,
            'llm_defer_prob': record.get('_llm_defer_prob'),
            'llm_defer_info': record.get('_llm_defer_info'),
            'llm_risk_score': record.get('_llm_risk_score'),
            'llm_risk_info': record.get('_llm_risk_info'),
            'bert_branch': record.get('bert_branch'),
            'gcn_branch': record.get('gcn_branch'),
            'branch_disagreement': _branch_disagreement(record, top1_id, top2_id) if top2_id is not None else None,
            'text': payload.get('text', '')
        })
    json.dump(stats, open(os.path.join(model_path, 'top2_correction_refinement.json'), 'w'), indent=4)
    # Also write to the old filename so downstream checks remain unchanged.
    json.dump(stats, open(os.path.join(model_path, 'llm_refinement.json'), 'w'), indent=4)
    print('Top-2 correction stats:', {k: v for k, v in stats.items() if k != 'records'})
    return predictions, stats


def calibrate_llm_risk_selector_from_dev(dev_confidence_records, dev_labels, rel_dict, args, model_path):
    """Build a soft dev error profile and optionally choose a target candidate rate."""
    global LLM_RISK_PROFILE
    paired = []
    profile_counts = {}
    pair_counts = {}
    for rec, gold in zip(dev_confidence_records, dev_labels):
        gold = int(gold)
        pred = int(rec.get('pred'))
        correct = pred == gold
        profile_key, pair_key, margin_bin = _risk_profile_keys(rec, rel_dict, args)
        for counts, key in [(profile_counts, profile_key), (pair_counts, pair_key)]:
            if key not in counts:
                counts[key] = {'total': 0, 'wrong': 0}
            counts[key]['total'] += 1
            counts[key]['wrong'] += 0 if correct else 1
        paired.append({'record': rec, 'gold': gold, 'correct': correct})

    n = len(paired)
    num_errors = sum(1 for x in paired if not x['correct'])
    global_error_rate = (num_errors / n) if n > 0 else 0.0
    LLM_RISK_PROFILE = {
        'global_error_rate': global_error_rate,
        'profile_counts': profile_counts,
        'pair_counts': pair_counts,
        'target_rate': None,
        'target_count_on_dev': None,
    }

    scored = []
    for x in paired:
        score, info = _llm_error_risk_score(x['record'], rel_dict, args)
        scored.append({
            'score': score,
            'correct': x['correct'],
            'gold': int(x['gold']),
            'pred': int(x['record'].get('pred')),
            'margin': float(x['record'].get('margin', 1.0)),
            'confidence': float(x['record'].get('confidence', 1.0)),
            'risk_info': info,
        })
    scored = sorted(scored, key=lambda x: (-x['score'], x['margin'], x['confidence']))

    max_k_by_rate = int(numpy.ceil(float(getattr(args, 'llm_dev_max_candidate_rate', 0.20)) * n)) if n > 0 else 0
    max_k_by_arg = int(getattr(args, 'llm_candidate_max_count', 30)) if int(getattr(args, 'llm_candidate_max_count', 30)) > 0 else n
    max_k = min(n, max(max_k_by_rate, int(getattr(args, 'llm_dev_min_candidates', 1))), max_k_by_arg)
    min_precision = float(getattr(args, 'llm_dev_min_candidate_error_precision', 0.40))
    chosen_k = 0
    chosen_precision = 0.0
    chosen_error_recall = 0.0
    prefix_stats = []
    if n > 0 and max_k > 0:
        for k in range(1, max_k + 1):
            prefix = scored[:k]
            wrong = sum(1 for z in prefix if not z['correct'])
            precision = wrong / k if k > 0 else 0.0
            recall = wrong / num_errors if num_errors > 0 else 0.0
            prefix_stats.append({'k': k, 'candidate_error_precision': precision, 'error_recall': recall})
            if precision >= min_precision:
                chosen_k = k
                chosen_precision = precision
                chosen_error_recall = recall
        if chosen_k == 0 and prefix_stats:
            best = max(prefix_stats, key=lambda z: (z['candidate_error_precision'], z['error_recall']))
            chosen_k = max(int(getattr(args, 'llm_dev_min_candidates', 1)), int(best['k']))
            chosen_k = min(chosen_k, max_k)
            chosen_precision = next((z['candidate_error_precision'] for z in prefix_stats if z['k'] == chosen_k), 0.0)
            chosen_error_recall = next((z['error_recall'] for z in prefix_stats if z['k'] == chosen_k), 0.0)

    target_rate = (chosen_k / n) if n > 0 and chosen_k > 0 else 0.0
    LLM_RISK_PROFILE['target_rate'] = target_rate
    LLM_RISK_PROFILE['target_count_on_dev'] = chosen_k

    stats = {
        'enabled': True,
        'num_dev_samples': n,
        'num_dev_errors': num_errors,
        'global_error_rate': global_error_rate,
        'risk_weights': {
            'margin': float(getattr(args, 'llm_margin_weight', 0.35)),
            'confidence': float(getattr(args, 'llm_conf_weight', 0.20)),
            'entropy': float(getattr(args, 'llm_entropy_weight', 0.25)),
            'profile': float(getattr(args, 'llm_profile_weight', 0.20)),
        },
        'target_count_on_dev': chosen_k,
        'target_rate': target_rate,
        'target_error_precision_on_dev': chosen_precision,
        'target_error_recall_on_dev': chosen_error_recall,
        'min_candidate_error_precision': min_precision,
        'max_candidate_rate': float(getattr(args, 'llm_dev_max_candidate_rate', 0.20)),
        'candidate_max_count': int(getattr(args, 'llm_candidate_max_count', 30)),
        'profile_counts': profile_counts,
        'pair_counts': pair_counts,
        'risk_rank_preview': scored[:50],
    }
    json.dump(stats, open(os.path.join(model_path, 'llm_dev_risk_selector.json'), 'w'), indent=4)
    print('Dev-calibrated LLM risk selector:')
    print(f"  dev samples/errors: {n}/{num_errors}")
    print(f"  chosen dev target count: {chosen_k}, target rate: {target_rate:.2%}")
    print(f"  candidate error precision on dev prefix: {chosen_precision:.2%}")
    print(f"  error recall on dev prefix: {chosen_error_recall:.2%}")
    return stats

def calibrate_llm_thresholds_from_dev(dev_confidence_records, dev_labels, args, model_path):
    """Calibrate the LLM candidate-selection margin threshold using dev-set errors.

    The purpose is not to use gold labels from the test set. We only use dev labels to learn
    a safer candidate region: samples with smaller top1-top2 margin are more likely to be
    wrong and therefore worth sending to the LLM. The calibrated threshold is chosen to
    cover a target recall of dev errors, while respecting a candidate-rate budget.
    """
    paired = []
    for rec, gold in zip(dev_confidence_records, dev_labels):
        pred = int(rec.get('pred'))
        gold = int(gold)
        margin = float(rec.get('margin', 1.0))
        conf = float(rec.get('confidence', 1.0))
        paired.append({
            'doc_id': rec.get('doc_id'),
            'epair': rec.get('epair'),
            'pred': pred,
            'gold': gold,
            'correct': pred == gold,
            'margin': margin,
            'confidence': conf,
        })

    n = len(paired)
    wrong = [x for x in paired if not x['correct']]
    stats = {
        'enabled': bool(args.llm_auto_threshold_from_dev),
        'num_dev_samples': n,
        'num_dev_errors': len(wrong),
        'dev_accuracy': (sum(1 for x in paired if x['correct']) / n) if n > 0 else 0.0,
        'original_candidate_filter': args.llm_candidate_filter,
        'original_margin_threshold': args.llm_margin_threshold,
        'original_conf_threshold': args.llm_conf_threshold,
        'target_error_recall': args.llm_dev_target_error_recall,
        'max_candidate_rate': args.llm_dev_max_candidate_rate,
        'min_candidates': args.llm_dev_min_candidates,
        'max_margin_threshold': args.llm_dev_max_margin_threshold,
    }

    if n == 0 or len(wrong) == 0:
        stats['status'] = 'no_dev_samples_or_no_dev_errors_keep_original_thresholds'
        json.dump(stats, open(os.path.join(model_path, 'llm_dev_threshold_calibration.json'), 'w'), indent=4)
        return stats

    sorted_all = sorted(paired, key=lambda x: (x['margin'], x['confidence']))
    wrong_sorted = sorted(wrong, key=lambda x: (x['margin'], x['confidence']))

    target_error_count = max(1, int(numpy.ceil(float(args.llm_dev_target_error_recall) * len(wrong_sorted))))
    target_error_count = min(target_error_count, len(wrong_sorted))
    threshold_by_error_recall = float(wrong_sorted[target_error_count - 1]['margin'])

    max_candidate_count = max(int(args.llm_dev_min_candidates), int(numpy.ceil(float(args.llm_dev_max_candidate_rate) * n)))
    max_candidate_count = min(max_candidate_count, n)
    threshold_by_budget = float(sorted_all[max_candidate_count - 1]['margin'])

    calibrated_margin_threshold = min(threshold_by_error_recall, threshold_by_budget, float(args.llm_dev_max_margin_threshold))
    calibrated_margin_threshold = max(0.0, calibrated_margin_threshold)

    # Evaluate the calibrated candidate region on dev.
    dev_candidates = [x for x in paired if x['margin'] <= calibrated_margin_threshold]
    dev_candidate_errors = [x for x in dev_candidates if not x['correct']]
    candidate_count = len(dev_candidates)
    candidate_error_count = len(dev_candidate_errors)
    candidate_rate = candidate_count / n if n > 0 else 0.0
    error_recall = candidate_error_count / len(wrong) if len(wrong) > 0 else 0.0
    candidate_error_precision = candidate_error_count / candidate_count if candidate_count > 0 else 0.0

    args.llm_margin_threshold = calibrated_margin_threshold
    if args.llm_dev_force_margin_filter:
        args.llm_candidate_filter = 'margin'

    stats.update({
        'status': 'calibrated',
        'threshold_by_error_recall': threshold_by_error_recall,
        'threshold_by_budget': threshold_by_budget,
        'calibrated_margin_threshold': calibrated_margin_threshold,
        'calibrated_candidate_filter': args.llm_candidate_filter,
        'dev_candidate_count': candidate_count,
        'dev_candidate_error_count': candidate_error_count,
        'dev_candidate_rate': candidate_rate,
        'dev_error_recall': error_recall,
        'dev_candidate_error_precision': candidate_error_precision,
        'dev_candidates_preview': [
            {
                'doc_id': x['doc_id'],
                'epair': list(x['epair']) if isinstance(x.get('epair'), tuple) else x.get('epair'),
                'margin': x['margin'],
                'confidence': x['confidence'],
                'pred': x['pred'],
                'gold': x['gold'],
                'correct': x['correct'],
            }
            for x in dev_candidates[:50]
        ]
    })

    print('Dev-calibrated LLM candidate threshold:')
    print(f"  dev samples/errors: {n}/{len(wrong)}")
    print(f"  margin_threshold: {calibrated_margin_threshold:.6f}")
    print(f"  dev candidates: {candidate_count} ({candidate_rate:.2%})")
    print(f"  dev error recall among candidates: {error_recall:.2%}")
    print(f"  dev candidate error precision: {candidate_error_precision:.2%}")

    json.dump(stats, open(os.path.join(model_path, 'llm_dev_threshold_calibration.json'), 'w'), indent=4)
    return stats

def apply_llm_refinement(predictions, confidence_records, rel_dict, args, model_path):
    """
    Refine low-confidence MulCo predictions using a small LLM at test time only.

    Important safety change:
    - In top2_judge mode, the LLM is not a free classifier.
    - It can only choose between the neural model's top-1 and top-2 labels.
    - The prediction is replaced only if the LLM selects top-2 and the original
      margin is very small. This prevents LLM from destroying high-quality base predictions.
    """
    candidates = []
    base_candidate_count_before_risk = 0
    risk_rank_all_count = 0
    risk_rank_target_count = None
    risk_rank_used_target_rate = None
    risk_rank_filtered_by_min_score = 0
    if getattr(args, 'llm_candidate_strategy', 'threshold') == 'defer_mlp':
        scored_candidates = []
        for record in confidence_records:
            payload = record.get('payload')
            if payload is None:
                continue
            if getattr(args, 'llm_branch_disagreement_only', False):
                top1_id, top2_id, _, _ = _top2_from_record(record, rel_dict)
                if not _branch_disagreement(record, top1_id, top2_id):
                    continue
            defer_prob, defer_info = predict_defer_probability(record, rel_dict, args)
            risk_score, risk_info = _llm_error_risk_score(record, rel_dict, args)
            record['_llm_defer_prob'] = defer_prob
            record['_llm_defer_info'] = defer_info
            record['_llm_risk_score'] = risk_score
            record['_llm_risk_info'] = risk_info
            scored_candidates.append(record)
        risk_rank_all_count = len(scored_candidates)
        scored_candidates = sorted(scored_candidates, key=lambda x: (-float(x.get('_llm_defer_prob', 0.0)), float(x.get('margin', 1.0)), float(x.get('confidence', 1.0))))
        if int(getattr(args, 'llm_candidate_target_count', 0)) > 0:
            target_count = int(args.llm_candidate_target_count)
        else:
            target_rate = (LLM_DEFER_PROFILE or {}).get('target_rate')
            if target_rate is not None and target_rate > 0:
                risk_rank_used_target_rate = float(target_rate)
                target_count = int(numpy.ceil(float(target_rate) * len(scored_candidates)))
            else:
                target_count = int(getattr(args, 'llm_candidate_max_count', 30))
        if int(getattr(args, 'llm_candidate_max_count', 0)) > 0:
            target_count = min(target_count, int(args.llm_candidate_max_count))
        if args.llm_max_cases and args.llm_max_cases > 0:
            target_count = min(target_count, int(args.llm_max_cases))
        target_count = max(0, min(target_count, len(scored_candidates)))
        risk_rank_target_count = target_count
        candidates = scored_candidates[:target_count]
        base_candidate_count_before_risk = len(scored_candidates)
    elif getattr(args, 'llm_candidate_strategy', 'threshold') == 'risk_rank':
        scored_candidates = []
        for record in confidence_records:
            payload = record.get('payload')
            if payload is None:
                continue
            top1_id, top2_id, _, _ = _top2_from_record(record, rel_dict)
            if getattr(args, 'llm_branch_disagreement_only', False) and not _branch_disagreement(record, top1_id, top2_id):
                continue
            risk_score, risk_info = _llm_error_risk_score(record, rel_dict, args)
            record['_llm_risk_score'] = risk_score
            record['_llm_risk_info'] = risk_info
            if risk_score < float(getattr(args, 'llm_min_risk_for_candidate', 0.0)):
                risk_rank_filtered_by_min_score += 1
                continue
            scored_candidates.append(record)
        risk_rank_all_count = len(scored_candidates)
        scored_candidates = sorted(scored_candidates, key=lambda x: (-float(x.get('_llm_risk_score', 0.0)), float(x.get('margin', 1.0)), float(x.get('confidence', 1.0))))
        if int(getattr(args, 'llm_candidate_target_count', 0)) > 0:
            target_count = int(args.llm_candidate_target_count)
        else:
            target_rate = (LLM_RISK_PROFILE or {}).get('target_rate')
            if target_rate is not None and target_rate > 0:
                risk_rank_used_target_rate = float(target_rate)
                target_count = int(numpy.ceil(float(target_rate) * len(scored_candidates)))
            else:
                target_count = int(getattr(args, 'llm_candidate_max_count', 30))
        if int(getattr(args, 'llm_candidate_max_count', 0)) > 0:
            target_count = min(target_count, int(args.llm_candidate_max_count))
        if args.llm_max_cases and args.llm_max_cases > 0:
            target_count = min(target_count, int(args.llm_max_cases))
        target_count = max(0, min(target_count, len(scored_candidates)))
        risk_rank_target_count = target_count
        candidates = scored_candidates[:target_count]
        base_candidate_count_before_risk = len(scored_candidates)
    else:
        for record in confidence_records:
            payload = record.get('payload')
            if payload is None:
                continue
            if args.llm_candidate_filter == 'and':
                use_candidate = record['confidence'] < args.llm_conf_threshold and record['margin'] < args.llm_margin_threshold
            elif args.llm_candidate_filter == 'margin':
                use_candidate = record['margin'] < args.llm_margin_threshold
            else:
                use_candidate = record['confidence'] < args.llm_conf_threshold or record['margin'] < args.llm_margin_threshold

            if use_candidate and getattr(args, 'llm_branch_disagreement_only', False):
                top1_id, top2_id, _, _ = _top2_from_record(record, rel_dict)
                use_candidate = _branch_disagreement(record, top1_id, top2_id)

            if use_candidate:
                risk_score, risk_info = _llm_error_risk_score(record, rel_dict, args)
                record['_llm_risk_score'] = risk_score
                record['_llm_risk_info'] = risk_info
                candidates.append(record)
        base_candidate_count_before_risk = len(candidates)
        candidates = sorted(candidates, key=lambda x: (x['margin'], x['confidence']))
        if args.llm_max_cases and args.llm_max_cases > 0:
            candidates = candidates[:args.llm_max_cases]

    stats = {
        'enabled': True,
        'mode': args.llm_refine_mode,
        'model_name': args.llm_model_name,
        'conf_threshold': args.llm_conf_threshold,
        'margin_threshold': args.llm_margin_threshold,
        'replace_margin_threshold': args.llm_replace_margin_threshold,
        'allow_llm_vague_replace': bool(args.allow_llm_vague_replace),
        'candidate_count': len(candidates),
        'changed_count': 0,
        'failed_count': 0,
        'kept_count': 0,
        'blocked_count': 0,
        'top2_choice_count': 0,
        'option1_choice_count': 0,
        'option2_choice_count': 0,
        'random_swap_enabled': not bool(args.llm_double_order_score),
        'double_order_enabled': bool(args.llm_double_order_score),
        'top2_score_method': args.llm_top2_score_method,
        'candidate_filter': args.llm_candidate_filter,
        'use_context_evidence': bool(args.llm_use_context_evidence),
        'use_verifier': bool(args.llm_use_verifier),
        'score_gap_threshold': args.llm_score_gap_threshold,
        'verify_delta': args.llm_verify_delta,
        'verify_top1_threshold': args.llm_verify_top1_threshold,
        'switch_delta': getattr(args, 'llm_switch_delta', None),
        'branch_disagreement_only': bool(getattr(args, 'llm_branch_disagreement_only', False)),
        'audit_only': bool(getattr(args, 'llm_audit_only', False)),
        'verify_then_top2_enabled': args.llm_refine_mode == 'verify_then_top2',
        'log_guided_ksu_enabled': args.llm_refine_mode == 'log_guided_ksu',
        'correction_mlp_enabled': args.llm_refine_mode == 'correction_mlp',
        'ksu_keep_count': 0,
        'ksu_switch_count': 0,
        'ksu_uncertain_count': 0,
        'audit_blocked_count': 0,
        'auto_threshold_from_dev': bool(getattr(args, 'llm_auto_threshold_from_dev', False)),
        'dev_calibrated_margin_threshold': args.llm_margin_threshold,
        'dev_calibrated_candidate_filter': args.llm_candidate_filter,
        'candidate_strategy': getattr(args, 'llm_candidate_strategy', 'threshold'),
        'defer_mlp_enabled': getattr(args, 'llm_candidate_strategy', 'threshold') == 'defer_mlp',
        'defer_mlp_target_rate': (LLM_DEFER_PROFILE or {}).get('target_rate'),
        'defer_mlp_dev_precision': (LLM_DEFER_PROFILE or {}).get('target_error_precision_on_dev'),
        'defer_mlp_dev_recall': (LLM_DEFER_PROFILE or {}).get('target_error_recall_on_dev'),
        'defer_mlp_margin_replace_gate': bool(getattr(args, 'llm_defer_use_margin_replace_gate', False)),
        'base_candidate_count_before_risk': base_candidate_count_before_risk,
        'risk_rank_all_count': risk_rank_all_count,
        'risk_rank_target_count': risk_rank_target_count,
        'risk_rank_used_target_rate': risk_rank_used_target_rate,
        'risk_rank_filtered_by_min_score': risk_rank_filtered_by_min_score,
        'risk_weights': {
            'margin': float(getattr(args, 'llm_margin_weight', 0.35)),
            'confidence': float(getattr(args, 'llm_conf_weight', 0.20)),
            'entropy': float(getattr(args, 'llm_entropy_weight', 0.25)),
            'profile': float(getattr(args, 'llm_profile_weight', 0.20)),
        },
        'min_risk_for_candidate': float(getattr(args, 'llm_min_risk_for_candidate', 0.0)),
        'min_risk_for_replace': float(getattr(args, 'llm_min_risk_for_replace', 0.0)),
        'verified_count': 0,
        'verify_blocked_count': 0,
        'score_gap_blocked_count': 0,
        'records': []
    }

    if len(candidates) == 0:
        print('LLM refinement enabled, but no low-confidence candidates were selected.')
        json.dump(stats, open(os.path.join(model_path, 'llm_refinement.json'), 'w'), indent=4)
        return predictions, stats

    if args.llm_refine_mode == 'correction_mlp':
        # No LLM is loaded in this mode. Candidate handoff is learned by the deferral MLP;
        # relationship re-selection is handled by a dev-trained top-2 correction network.
        return apply_top2_correction_refinement(predictions, candidates, rel_dict, args, model_path, stats)

    try:
        device = torch.device('cuda' if args.llm_device == 'cuda' and torch.cuda.is_available() else 'cpu')
        print(f'Loading LLM refinement model: {args.llm_model_name} on {device}')
        llm_config = AutoConfig.from_pretrained(args.llm_model_name, trust_remote_code=True)
        llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name, trust_remote_code=True)

        llm_is_encoder_decoder = bool(getattr(llm_config, 'is_encoder_decoder', False))
        if llm_is_encoder_decoder:
            llm_model = AutoModelForSeq2SeqLM.from_pretrained(args.llm_model_name, trust_remote_code=True)
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_name, trust_remote_code=True)
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token

        if device.type == 'cuda':
            llm_model = llm_model.half()
        llm_model = llm_model.to(device)
        llm_model.eval()
    except Exception as exc:
        print('Failed to load LLM refinement model. Keeping original MulCo predictions.')
        print('LLM load error:', repr(exc))
        stats['enabled'] = False
        stats['failed_count'] = len(candidates)
        stats['error'] = repr(exc)
        json.dump(stats, open(os.path.join(model_path, 'llm_refinement.json'), 'w'), indent=4)
        return predictions, stats

    id_to_rel = _id_to_relation(rel_dict)
    with torch.no_grad():
        for record in tqdm(candidates, desc='LLM refining low-confidence pairs'):
            payload = record['payload']
            base_pred = int(record['pred'])
            top1_id, top2_id, top1_prob, top2_prob = _top2_from_record(record, rel_dict)
            top1_id = int(top1_id)
            top2_id = int(top2_id) if top2_id is not None else None
            decision_reason = 'not_processed'
            llm_pred = None
            llm_choice = None
            final_pred = base_pred

            option1_id = None
            option2_id = None
            option1_source = None
            option2_source = None
            llm_score_gap = None
            verify_top1_score = None
            verify_top2_score = None
            verify_top1_raw = None
            verify_top2_raw = None
            double_order_details = {}

            if args.llm_refine_mode == 'log_guided_ksu':
                if top2_id is None:
                    stats['failed_count'] += 1
                    stats['records'].append({
                        'doc_id': record['doc_id'],
                        'epair': list(record['epair']),
                        'confidence': record['confidence'],
                        'margin': record['margin'],
                        'base_pred': id_to_rel.get(base_pred, str(base_pred)),
                        'final_pred': id_to_rel.get(final_pred, str(final_pred)),
                        'decision': 'no_top2',
                        'text': payload.get('text', '')
                    })
                    continue

                proposed_pred = int(top2_id)
                proposed_key = id_to_rel.get(proposed_pred, str(proposed_pred))
                prompt = build_llm_log_guided_ksu_prompt(payload, rel_dict, record, top1_id, top2_id)
                llm_choice, ksu_scores = score_ksu_choice(
                    llm_model, llm_tokenizer, prompt, device,
                    llm_is_encoder_decoder, args.llm_max_input_length
                )
                llm_score_1 = ksu_scores.get('K')
                llm_score_2 = ksu_scores.get('S')
                llm_score_gap = None
                if llm_score_1 is not None and llm_score_2 is not None:
                    llm_score_gap = llm_score_2 - llm_score_1
                answer = f'log_guided_ksu: choice={llm_choice}, scores={ksu_scores}, switch_minus_keep={llm_score_gap}'

                if llm_choice == 'K':
                    stats['kept_count'] += 1
                    stats['ksu_keep_count'] += 1
                    decision_reason = 'ksu_keep_top1'
                    final_pred = base_pred
                elif llm_choice == 'U':
                    stats['blocked_count'] += 1
                    stats['ksu_uncertain_count'] += 1
                    decision_reason = 'ksu_uncertain_keep_top1'
                    final_pred = base_pred
                elif llm_choice == 'S':
                    stats['ksu_switch_count'] += 1
                    if getattr(args, 'llm_audit_only', False):
                        stats['blocked_count'] += 1
                        stats['audit_blocked_count'] += 1
                        decision_reason = 'audit_only_switch_not_applied'
                        final_pred = base_pred
                    elif proposed_key == 'v' and not args.allow_llm_vague_replace:
                        stats['blocked_count'] += 1
                        decision_reason = 'blocked_vague_replacement'
                        final_pred = base_pred
                    elif float(record.get('_llm_risk_score', 0.0)) < float(getattr(args, 'llm_min_risk_for_replace', 0.0)):
                        stats['blocked_count'] += 1
                        decision_reason = 'blocked_risk_too_low'
                        final_pred = base_pred
                    elif (not (getattr(args, 'llm_candidate_strategy', 'threshold') == 'defer_mlp' and not getattr(args, 'llm_defer_use_margin_replace_gate', False))) and record['margin'] > args.llm_replace_margin_threshold:
                        stats['blocked_count'] += 1
                        decision_reason = 'blocked_margin_too_large'
                        final_pred = base_pred
                    elif llm_score_gap is not None and llm_score_gap < args.llm_switch_delta:
                        stats['blocked_count'] += 1
                        stats['score_gap_blocked_count'] += 1
                        decision_reason = 'blocked_switch_delta_too_small'
                        final_pred = base_pred
                    else:
                        final_pred = proposed_pred
                        decision_reason = 'replaced_by_top2_log_guided_ksu'
                        stats['top2_choice_count'] += 1
                else:
                    stats['failed_count'] += 1
                    decision_reason = 'ksu_score_failed_keep_top1'
                    final_pred = base_pred
                llm_pred = None

            elif args.llm_refine_mode == 'verify_then_top2':
                if top2_id is None:
                    stats['failed_count'] += 1
                    stats['records'].append({
                        'doc_id': record['doc_id'],
                        'epair': list(record['epair']),
                        'confidence': record['confidence'],
                        'margin': record['margin'],
                        'base_pred': id_to_rel.get(base_pred, str(base_pred)),
                        'final_pred': id_to_rel.get(final_pred, str(final_pred)),
                        'decision': 'no_top2',
                        'text': payload.get('text', '')
                    })
                    continue

                # Verifier-corrector mode:
                # Do not ask the LLM to directly choose a label. Instead, score whether
                # the current top1 and alternative top2 relations are supported by evidence.
                top1_verify_prompt = build_llm_relation_verifier_prompt(payload, rel_dict, top1_id)
                top2_verify_prompt = build_llm_relation_verifier_prompt(payload, rel_dict, top2_id)
                verify_top1_score, verify_top1_raw = score_binary_verdict(
                    llm_model, llm_tokenizer, top1_verify_prompt, device,
                    llm_is_encoder_decoder, args.llm_max_input_length
                )
                verify_top2_score, verify_top2_raw = score_binary_verdict(
                    llm_model, llm_tokenizer, top2_verify_prompt, device,
                    llm_is_encoder_decoder, args.llm_max_input_length
                )
                stats['verified_count'] += 1

                proposed_pred = int(top2_id)
                proposed_key = id_to_rel.get(proposed_pred, str(proposed_pred))
                llm_score_gap = None
                if verify_top1_score is not None and verify_top2_score is not None:
                    llm_score_gap = verify_top2_score - verify_top1_score

                if verify_top1_score is None or verify_top2_score is None:
                    stats['failed_count'] += 1
                    decision_reason = 'verifier_failed_keep_top1'
                    final_pred = base_pred
                elif verify_top1_score > args.llm_verify_top1_threshold:
                    stats['kept_count'] += 1
                    decision_reason = 'top1_supported_keep'
                    final_pred = base_pred
                elif (verify_top2_score - verify_top1_score) < args.llm_verify_delta:
                    stats['blocked_count'] += 1
                    stats['verify_blocked_count'] += 1
                    decision_reason = 'blocked_top2_not_enough_better'
                    final_pred = base_pred
                elif record['margin'] > args.llm_replace_margin_threshold:
                    stats['blocked_count'] += 1
                    decision_reason = 'blocked_margin_too_large'
                    final_pred = base_pred
                elif proposed_key == 'v' and not args.allow_llm_vague_replace:
                    stats['blocked_count'] += 1
                    decision_reason = 'blocked_vague_replacement'
                    final_pred = base_pred
                else:
                    final_pred = proposed_pred
                    decision_reason = 'replaced_by_top2_verified_corrector'
                    stats['top2_choice_count'] += 1

                answer = (
                    f'verify_then_top2: top1_score={verify_top1_score}, top2_score={verify_top2_score}, '
                    f'top2_minus_top1={llm_score_gap}, '
                    f'top1_raw={verify_top1_raw}, top2_raw={verify_top2_raw}'
                )
                llm_choice = None
                llm_score_1 = verify_top1_score
                llm_score_2 = verify_top2_score

            elif args.llm_refine_mode == 'top2_judge':
                if top2_id is None:
                    stats['failed_count'] += 1
                    stats['records'].append({
                        'doc_id': record['doc_id'],
                        'epair': list(record['epair']),
                        'confidence': record['confidence'],
                        'margin': record['margin'],
                        'base_pred': id_to_rel.get(base_pred, str(base_pred)),
                        'final_pred': id_to_rel.get(final_pred, str(final_pred)),
                        'decision': 'no_top2',
                        'text': payload.get('text', '')
                    })
                    continue

                if args.llm_double_order_score:
                    # In double-order mode, records use normalized options:
                    # option1/top1 and option2/top2. The actual LLM scoring still sees both orders.
                    option1_id, option2_id = top1_id, top2_id
                    option1_source, option2_source = 'top1', 'top2'
                    prompt = None
                else:
                    # Blind randomized top-2 order:
                    # do not let the LLM know which candidate is the neural model top-1.
                    if random.random() < 0.5:
                        option1_id, option2_id = top1_id, top2_id
                        option1_source, option2_source = 'top1', 'top2'
                    else:
                        option1_id, option2_id = top2_id, top1_id
                        option1_source, option2_source = 'top2', 'top1'

                    if args.llm_use_context_evidence:
                        prompt = build_llm_context_top2_judge_prompt(payload, rel_dict, option1_id, option2_id)
                    else:
                        prompt = build_llm_top2_judge_prompt(payload, rel_dict, option1_id, option2_id)
            else:
                prompt = build_llm_prompt(payload, rel_dict)

            try:
                if args.llm_refine_mode != 'log_guided_ksu':
                    llm_score_1 = None
                    llm_score_2 = None
                if args.llm_refine_mode == 'log_guided_ksu':
                    # K/S/U mode has already computed scores and made the conservative decision above.
                    pass
                elif args.llm_refine_mode == 'verify_then_top2':
                    # Verifier mode has already computed verify_top1_score / verify_top2_score above.
                    # Do not call generate() here; otherwise `prompt` is undefined and every sample fails.
                    llm_score_1 = verify_top1_score
                    llm_score_2 = verify_top2_score
                elif args.llm_refine_mode == 'top2_judge' and args.llm_top2_score_method == 'token_score':
                    if args.llm_double_order_score:
                        llm_choice, llm_scores, double_order_details = score_top2_double_order_choice(
                            llm_model,
                            llm_tokenizer,
                            payload,
                            rel_dict,
                            top1_id,
                            top2_id,
                            device,
                            llm_is_encoder_decoder,
                            args.llm_max_input_length,
                            use_context_evidence=args.llm_use_context_evidence
                        )
                        llm_score_1 = llm_scores.get(1) if isinstance(llm_scores, dict) else None
                        llm_score_2 = llm_scores.get(2) if isinstance(llm_scores, dict) else None
                        answer = (
                            f'double_order_token_score: top1_score={llm_score_1}, top2_score={llm_score_2}, '
                            f'details={double_order_details}'
                        )
                    else:
                        llm_choice, llm_scores = score_top2_choice(
                            llm_model,
                            llm_tokenizer,
                            prompt,
                            device,
                            llm_is_encoder_decoder,
                            args.llm_max_input_length
                        )
                        llm_score_1 = llm_scores.get(1) if isinstance(llm_scores, dict) else None
                        llm_score_2 = llm_scores.get(2) if isinstance(llm_scores, dict) else None
                        answer = f'token_score: score_1={llm_score_1}, score_2={llm_score_2}'
                else:
                    encoded = llm_tokenizer(
                        prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=args.llm_max_input_length
                    ).to(device)
                    generated = llm_model.generate(
                        **encoded,
                        max_new_tokens=args.llm_max_new_tokens,
                        do_sample=False,
                        pad_token_id=llm_tokenizer.pad_token_id
                    )
                    if llm_is_encoder_decoder:
                        answer_ids = generated[0]
                    else:
                        answer_ids = generated[0][encoded['input_ids'].shape[-1]:]
                    answer = llm_tokenizer.decode(answer_ids, skip_special_tokens=True)

                if args.llm_refine_mode == 'log_guided_ksu':
                    pass
                elif args.llm_refine_mode == 'top2_judge':
                    if args.llm_top2_score_method != 'token_score':
                        llm_choice = parse_llm_top2_choice(answer)
                    if llm_choice is None:
                        stats['failed_count'] += 1
                        decision_reason = 'score_or_parse_failed_keep_top1'
                        final_pred = base_pred
                    else:
                        if llm_choice == 1:
                            stats['option1_choice_count'] += 1
                            chosen_pred = int(option1_id)
                            chosen_source = option1_source
                        else:
                            stats['option2_choice_count'] += 1
                            chosen_pred = int(option2_id)
                            chosen_source = option2_source

                        if chosen_source == 'top2':
                            stats['top2_choice_count'] += 1

                        if chosen_pred == top1_id:
                            stats['kept_count'] += 1
                            decision_reason = 'llm_chose_top1_keep'
                            final_pred = base_pred
                        else:
                            proposed_pred = chosen_pred
                            proposed_key = id_to_rel.get(proposed_pred, str(proposed_pred))

                            # Option score gap protection: the LLM must strongly prefer the proposed top2 option.
                            if llm_score_1 is not None and llm_score_2 is not None:
                                chosen_score = llm_score_1 if llm_choice == 1 else llm_score_2
                                other_score = llm_score_2 if llm_choice == 1 else llm_score_1
                                llm_score_gap = chosen_score - other_score
                            else:
                                llm_score_gap = None

                            # Run evidence-aware verifier BEFORE margin/score-gap blocking so that
                            # llm_refinement.json records whether the proposed top2 is actually better supported.
                            verifier_passed = True
                            if args.llm_use_verifier:
                                top1_verify_prompt = build_llm_relation_verifier_prompt(payload, rel_dict, top1_id)
                                top2_verify_prompt = build_llm_relation_verifier_prompt(payload, rel_dict, proposed_pred)
                                verify_top1_score, verify_top1_raw = score_binary_verdict(
                                    llm_model, llm_tokenizer, top1_verify_prompt, device,
                                    llm_is_encoder_decoder, args.llm_max_input_length
                                )
                                verify_top2_score, verify_top2_raw = score_binary_verdict(
                                    llm_model, llm_tokenizer, top2_verify_prompt, device,
                                    llm_is_encoder_decoder, args.llm_max_input_length
                                )
                                stats['verified_count'] += 1

                                if verify_top1_score is None or verify_top2_score is None:
                                    verifier_passed = False
                                    stats['blocked_count'] += 1
                                    stats['verify_blocked_count'] += 1
                                    decision_reason = 'blocked_verifier_failed'
                                    final_pred = base_pred
                                elif verify_top1_score > args.llm_verify_top1_threshold:
                                    verifier_passed = False
                                    stats['blocked_count'] += 1
                                    stats['verify_blocked_count'] += 1
                                    decision_reason = 'blocked_top1_still_supported'
                                    final_pred = base_pred
                                elif (verify_top2_score - verify_top1_score) < args.llm_verify_delta:
                                    verifier_passed = False
                                    stats['blocked_count'] += 1
                                    stats['verify_blocked_count'] += 1
                                    decision_reason = 'blocked_verifier_delta_too_small'
                                    final_pred = base_pred

                            if verifier_passed:
                                if record['margin'] > args.llm_replace_margin_threshold:
                                    stats['blocked_count'] += 1
                                    decision_reason = 'blocked_margin_too_large'
                                    final_pred = base_pred
                                elif proposed_key == 'v' and not args.allow_llm_vague_replace:
                                    stats['blocked_count'] += 1
                                    decision_reason = 'blocked_vague_replacement'
                                    final_pred = base_pred
                                elif llm_score_gap is not None and llm_score_gap < args.llm_score_gap_threshold:
                                    stats['blocked_count'] += 1
                                    stats['score_gap_blocked_count'] += 1
                                    decision_reason = 'blocked_llm_score_gap_too_small'
                                    final_pred = base_pred
                                else:
                                    final_pred = proposed_pred
                                    decision_reason = 'replaced_by_top2_context_verified' if args.llm_use_verifier else 'replaced_by_top2'
                elif args.llm_refine_mode == 'verify_then_top2':
                    # Decision was already made by verifier-corrector logic above.
                    pass
                else:
                    llm_pred = parse_llm_label(answer, rel_dict)
                    if llm_pred is None:
                        stats['failed_count'] += 1
                        decision_reason = 'parse_failed_keep_base'
                        final_pred = base_pred
                    else:
                        proposed_key = id_to_rel.get(int(llm_pred), str(llm_pred))
                        if proposed_key == 'v' and not args.allow_llm_vague_replace:
                            stats['blocked_count'] += 1
                            decision_reason = 'blocked_vague_replacement'
                            final_pred = base_pred
                        else:
                            final_pred = int(llm_pred)
                            decision_reason = 'free_replaced_or_kept'

                if final_pred != base_pred:
                    doc_id = record['doc_id']
                    epair = record['epair']
                    predictions[doc_id][epair] = final_pred
                    stats['changed_count'] += 1

                stats['records'].append({
                    'doc_id': record['doc_id'],
                    'epair': list(record['epair']),
                    'confidence': record['confidence'],
                    'margin': record['margin'],
                    'top1_prob': top1_prob,
                    'top2_prob': top2_prob,
                    'top1_label': id_to_rel.get(top1_id, str(top1_id)),
                    'top2_label': id_to_rel.get(top2_id, str(top2_id)) if top2_id is not None else None,
                    'option1_label': id_to_rel.get(option1_id, str(option1_id)) if option1_id is not None else None,
                    'option2_label': id_to_rel.get(option2_id, str(option2_id)) if option2_id is not None else None,
                    'option1_source': option1_source,
                    'option2_source': option2_source,
                    'base_pred': id_to_rel.get(base_pred, str(base_pred)),
                    'llm_answer': answer,
                    'llm_choice': llm_choice,
                    'llm_score_1': llm_score_1 if 'llm_score_1' in locals() else None,
                    'llm_score_2': llm_score_2 if 'llm_score_2' in locals() else None,
                    'llm_score_gap': llm_score_gap,
                    'double_order_details': double_order_details,
                    'bert_branch': record.get('bert_branch'),
                    'gcn_branch': record.get('gcn_branch'),
                    'branch_disagreement': _branch_disagreement(record, top1_id, top2_id) if top2_id is not None else None,
                    'llm_risk_score': record.get('_llm_risk_score'),
                    'llm_risk_info': record.get('_llm_risk_info'),
                'llm_defer_prob': record.get('_llm_defer_prob'),
                'llm_defer_info': record.get('_llm_defer_info'),
                    'verify_top1_score': verify_top1_score,
                    'verify_top2_score': verify_top2_score,
                    'verify_top1_raw': verify_top1_raw,
                    'verify_top2_raw': verify_top2_raw,
                    'llm_pred': id_to_rel.get(llm_pred, str(llm_pred)) if llm_pred is not None else None,
                    'final_pred': id_to_rel.get(final_pred, str(final_pred)),
                    'decision': decision_reason,
                    'text': payload.get('text', '')
                })
            except Exception as exc:
                stats['failed_count'] += 1
                stats['records'].append({
                    'doc_id': record['doc_id'],
                    'epair': list(record['epair']),
                    'confidence': record['confidence'],
                    'margin': record['margin'],
                    'base_pred': id_to_rel.get(base_pred, str(base_pred)),
                    'error': repr(exc),
                    'decision': 'exception_keep_base',
                    'text': payload.get('text', '')
                })

    json.dump(stats, open(os.path.join(model_path, 'llm_refinement.json'), 'w'), indent=4)
    print('LLM refinement stats:', {k: v for k, v in stats.items() if k != 'records'})
    return predictions, stats


def compute_scores(preds, gold, label_vocab, ignore_files=[], distance_flags=None):
    # print(gold)
    # print(preds)
    label_reverse = {'i': 'ii', 'ii': 'i', 'b': 'a', 'a': 'b', 'm': 'im', 'im': 'm', 's': 's', 'v': 'v'}
    gold_count = 0.0
    pred_count = 0.0
    correct_count = 0.0
    short_distance_gold = 0.0
    short_distance_pred = 0.0
    short_distance_correct = 0.0
    long_distance_gold = 0.0
    long_distance_pred = 0.0
    long_distance_correct = 0.0
    for file in preds:
        if ignore_files and file in ignore_files:
            print('Ignoring {}'.format(file))
            continue
        for epair in gold[file]:
            gold_count += 1
            if distance_flags is not None:
                if distance_flags[file][epair]:
                    short_distance_gold += 1
                else:
                    long_distance_gold += 1
            if epair not in preds[file] and (epair[1], epair[0]) not in preds[file]:
                continue
            pred_count += 1
            if distance_flags is not None:
                if distance_flags[file][epair]:
                    short_distance_pred += 1
                else:
                    long_distance_pred += 1
            if epair in preds[file]:
                if preds[file][epair] == gold[file][epair]:
                    correct_count += 1
                    if distance_flags is not None:
                        if distance_flags[file][epair]:
                            short_distance_correct += 1
                        else:
                            long_distance_correct += 1
            elif (epair[1], epair[0]) in preds[file]:
                rev_epair = (epair[1], epair[0])
                pred_label = ''
                for key in label_vocab:
                    if label_vocab[key] == preds[file][rev_epair]:
                        pred_label = key
                rev_pred_label = label_vocab[label_reverse[pred_label]]
                if rev_pred_label == gold[file][epair]:
                    correct_count += 1
                    if distance_flags is not None:
                        if distance_flags[file][epair]:
                            short_distance_correct += 1
                        else:
                            long_distance_correct += 1

    precision = correct_count / pred_count if pred_count != 0.0 else 0.0
    recall = correct_count / gold_count if gold_count != 0.0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0.0 else 0.0
    short_precision = short_distance_correct / short_distance_pred if short_distance_pred != 0.0 else 0.0
    short_recall = short_distance_correct / short_distance_gold if short_distance_gold != 0.0 else 0.0
    short_f1 = (2 * short_precision * short_recall) / (
                short_precision + short_recall) if short_precision + short_recall != 0.0 else 0.0
    long_precision = long_distance_correct / long_distance_pred if long_distance_pred != 0.0 else 0.0
    long_recall = long_distance_correct / long_distance_gold if long_distance_gold != 0.0 else 0.0
    long_f1 = (2 * long_precision * long_recall) / (
                long_precision + long_recall) if long_precision + long_recall != 0.0 else 0.0

    results = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ShortDistancePrecision": short_precision,
        "ShortDistanceRecall": short_recall,
        "ShortDistanceF1": short_f1,
        "LongDistancePrecision": long_precision,
        "LongDistanceRecall": long_recall,
        "LongDistanceF1": long_f1
    }

    return results


def to_torch_syntax_graph(graph, graph_type="syntax"):
    if graph_type == "syntax":
        type_dict = syn_edge_type_dict

    else:
        type_dict = temporal_edge_type_dict

    nodes = set()
    edges = []

    for edge in graph.edge_list:
        if hasattr(edge, 'orig_node'):
            edges.append((edge.orig_node, edge.targ_node, type_dict[edge.relation]))
            nodes.update([edge.orig_node, edge.targ_node])

        else:
            edges.append((edge.node_one, edge.node_two, type_dict[edge.relation]))
            nodes.update([edge.node_one, edge.node_two])

    if graph_type == "syntax":
        nodes = sorted([n for n in nodes if "D" in n]) + sorted([n for n in nodes if "v" in n],
                                                                key=lambda x: int(x[1:])) + sorted(
            [n for n in nodes if "w" in n], key=lambda x: int(x[1:]))

    else:
        nodes = sorted([n for n in nodes if "D" in n]) + sorted([n for n in nodes if "ei" in n],
                                                                key=lambda x: int(x[2:])) + sorted(
            [n for n in nodes if "t" in n], key=lambda x: int(x[1:]))

    graph = networkx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    graph_torch = torch_geometric.utils.from_networkx(graph)
    graph_torch.nodes = list(graph.nodes)
    graph_torch.node_feat = None
    graph_torch.edge_type = graph_torch.weight
    graph_torch.node_idx = {node: i for i, node in enumerate(graph.nodes)}

    return graph_torch


@torch.no_grad()
def get_node_feat(encoder, graph, tokenized_doc_sents):
    sent_nums = tokenized_doc_sents["input_ids"].shape[0]
    mappings = [tokenized_doc_sents.word_ids(i) for i in range(sent_nums)]
    tokenized_inputs = {x: y for x, y in tokenized_doc_sents.items()}
    output = encoder(**tokenized_inputs, output_hidden_states=True)
    last_four_layer_embeds = output.hidden_states[-2:]
    token_embeds = torch.mean(torch.vstack([x.unsqueeze(0) for x in last_four_layer_embeds]), dim=0)

    sent_embeds = []
    node_feats = []

    for i in range(sent_nums):
        sent_tokens = token_embeds[i]
        mapping = [m if m != None else -1 for m in mappings[i]]
        mapping = [m if m >= 0 else max(mapping) + 1 for m in mapping]
        index = torch.Tensor(mapping).long()
        node_feat = scatter_mean(sent_tokens, index, dim=0)[:-1]
        sent_embeds.append(torch.mean(node_feat, dim=0))
        node_feats.append(node_feat)

    sent_embeds = torch.vstack(sent_embeds)
    doc_embed = torch.mean(sent_embeds, dim=0).unsqueeze(0)

    node_feats = torch.vstack(node_feats)
    final_node_feat = torch.vstack([doc_embed, sent_embeds, node_feats])
    torch.cuda.empty_cache()

    return final_node_feat


def event_re_mapping(sents, timex):
    offset = 0
    sent_offset = []

    for sent in sents:
        sent_offset.append(offset)
        offset += len(sent)

    timex_word_mapping = {}

    for t_id, (sent_num, pos, tokens, _) in timex.items():
        tokens_len = len(tokens.split())
        timex_word_mapping[t_id] = [sent_offset[int(sent_num)] + pos, sent_offset[int(sent_num)] + pos + tokens_len]

    return timex_word_mapping


if __name__ == '__main__':
    if args.bert_model == "bert":
        bert_model_name = "bert-base-uncased"

    elif args.bert_model == "roberta":
        bert_model_name = "roberta-base"

    else:
        bert_model_name = "bert-base-uncased"

    model_path = os.path.join("./models", str(abs(hash(str(vars(args)))) % 10000))
    cache_dir = os.path.join("cache-%s" % (args.bert_model), args.gold_pairs)

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    test_files = ['APW19980418.0210', 'CNN19980213.2130.0155', 'APW19980227.0494', \
                  'CNN19980126.1600.1104', 'NYT19980402.0453', 'APW19980227.0489', \
                  'PRI19980306.2000.1675', 'PRI19980115.2000.0186', 'APW19980308.0201']
    dev_files = ['APW19980227.0487', 'PRI19980216.2000.0170', 'ed980111.1130.0089', \
                 'CNN19980223.1130.0960', 'NYT19980212.0019']

    if "tbd" in args.gold_pairs or "tdd" in args.gold_pairs:
        rel_dict = {'a': 0, 'b': 1, 's': 2, 'i': 3, 'ii': 4, 'v': 5}
        label_reverse = {'a': 'b', 'b': 'a', 'i': 'ii', 'ii': 'i', 's': 's', 'v': 'v'}

    else:
        rel_dict = {'a': 0, 'b': 1, 'e': 2, 'v': 3}
        label_reverse = {'a': 'b', 'b': 'a', 'e': 'e', 'v': 'v'}

    label_reverse_map = {rel_dict[label]: rel_dict[reverse_label] for label, reverse_label in label_reverse.items()}
    # Conservative transitivity rules: before+before->before and after+after->after.
    # MATRES/TBD both use 'a' and 'b'; other labels are left untouched to avoid noisy rules.
    transitive_label_ids = [rel_dict[label] for label in ['a', 'b'] if label in rel_dict]

    if args.bert_model == "roberta":
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name, add_prefix_space=True)

    else:
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    bert_config = AutoConfig.from_pretrained(bert_model_name)
    bert_encoder = AutoModel.from_pretrained(bert_model_name, config=bert_config)

    full_dict = pickle.load(open(args.doc_file, 'rb'))
    annotated_event_pairs = pickle.load(open(args.gold_pairs, 'rb'))
    test_event_pairs = pickle.load(open(args.test_pairs, 'rb'))
    event_map = pickle.load(open(args.event_map, 'rb'))
    syntax_graphs = pickle.load(open(args.syntax_file, 'rb'))
    temporal_graphs = pickle.load(open(args.temporal_file, 'rb'))
    timex_dict = pickle.load(open('timex_dict.pkl', 'rb'))

    train_data = []
    dev_data = []
    test_data = []
    gold_test_labels = {}
    distance_flags = {}

    for document in tqdm(full_dict, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        doc_id = document.split('.tml')[0]
        distance_flags[doc_id] = {}
        # Since some documents are not present in TDD-Man, those files will be skipped
        if doc_id not in annotated_event_pairs:
            continue

        doc_dict = full_dict[document]
        doc_sents = doc_dict["sents"]
        doc_sents = {int(x): y for x, y in doc_sents.items()}
        doc_events = doc_dict["events"]
        doc_events = {x: [int(y[0]), int(y[1]) - 1, y[2]] for x, y in doc_events.items()}

        if "hand" not in args.temporal_file:
            doc_timexes = timex_dict[doc_id]
            doc_timexes = {x: [int(y[0]), int(y[1]), y[2], y[3]] for x, y in doc_timexes.items()}

        else:
            doc_timexes = doc_dict["timexes"]
            doc_timexes = {x: [int(y[0]), int(y[1]) - 1, y[2], y[3]] for x, y in doc_timexes.items()}

        max_sent_num = max(list(doc_sents.keys()))
        word_idx_to_loc = []
        loc_to_word_idx = {}
        event_to_word_idx = {}
        sents = []

        for i, sent in sorted(doc_sents.items(), key=lambda x: x[0]):
            tokens = sent.split()
            sents.append(tokens)

            for j, token in enumerate(tokens):
                word_idx_to_loc.append((i, j))

        for i, loc in enumerate(word_idx_to_loc):
            loc_to_word_idx[loc] = i

        for e, loc in doc_events.items():
            sent, offset, _ = loc
            event_to_word_idx[e] = loc_to_word_idx[(sent, offset)] + len(doc_sents) + 1

        for e, loc in doc_timexes.items():
            sent, offset, _, _ = loc
            event_to_word_idx[e] = loc_to_word_idx[(sent, offset)] + len(doc_sents) + 1

        syntax_graph = to_torch_syntax_graph(syntax_graphs[doc_id], graph_type="syntax")
        tokenized_doc_sents = tokenizer(
            sents,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            max_length=512,
            return_tensors='pt'
        )

        try:
            if args.re_init:
                1 / 0
            node_feat = torch.load(os.path.join(cache_dir, "%s-graph.pkl" % (doc_id)))

        except:
            node_feat = get_node_feat(bert_encoder, syntax_graph, tokenized_doc_sents)
            torch.save(node_feat, os.path.join(cache_dir, "%s-graph.pkl" % (doc_id)))

        syntax_graph.node_feat = node_feat
        syntax_graph.tokenized_doc_sents = tokenized_doc_sents
        temporal_graph = to_torch_syntax_graph(temporal_graphs[doc_id], graph_type="temporal")
        temporal_graph.tokenized_doc_sents = tokenized_doc_sents
        temporal_graph.event_to_word_idx = {e: i for e, i in event_to_word_idx.items() if e in temporal_graph.nodes}
        temporal_graph.node_feat = torch.vstack([syntax_graph.node_feat[0]] + [syntax_graph.node_feat[i] for e, i in
                                                                               temporal_graph.event_to_word_idx.items()])
        syntax_graph.event_to_word_idx = temporal_graph.event_to_word_idx
        current_event_pairs = annotated_event_pairs[doc_id]

        if doc_id in test_files:
            gold_test_labels[doc_id] = {}
            for epair in current_event_pairs:
                e1, e2, label = epair
                if e1.startswith('t') or e2.startswith('t'):
                    continue
                if label == 'v':
                    continue
                gold_test_labels[doc_id][(event_map[doc_id][e1], event_map[doc_id][e2])] = rel_dict[label]
                e1_sent_index = doc_events[event_map[doc_id][e1]][0]
                e2_sent_index = doc_events[event_map[doc_id][e2]][0]
                distance_flags[doc_id][(event_map[doc_id][e1], event_map[doc_id][e2])] = True \
                    if abs(e1_sent_index - e2_sent_index) < args.dist_cutoff else False
            current_event_pairs = test_event_pairs[doc_id]

        for epair in current_event_pairs:
            e1, e2, label = epair
            if label == 'v':
                continue
            if e1 not in event_map[doc_id] or e2 not in event_map[doc_id]:
                continue
            if event_map[doc_id][e1] not in doc_events or event_map[doc_id][e2] not in doc_events:
                continue
            e1_data = doc_events[event_map[doc_id][e1]]
            e2_data = doc_events[event_map[doc_id][e2]]
            e1_sent_index = e1_data[0]
            e2_sent_index = e2_data[0]
            distance_flags[doc_id][(event_map[doc_id][e1], event_map[doc_id][e2])] = True \
                if abs(e1_sent_index - e2_sent_index) < args.dist_cutoff else False
            if abs(e1_sent_index - e2_sent_index) > 15 and doc_id in test_files:
                continue

            if args.dist_type != "mix":
                if args.dist_type == "short" and not distance_flags[doc_id][
                    (event_map[doc_id][e1], event_map[doc_id][e2])]:
                    continue

                if args.dist_type == "long" and distance_flags[doc_id][(event_map[doc_id][e1], event_map[doc_id][e2])]:
                    continue

            min_sent = max(0, min([e1_sent_index, e2_sent_index]) - 1)
            min_token_index = 0

            for i in range(0, min_sent):
                min_token_index += len(doc_sents[i].split())

            max_sent = min(max_sent_num, max([e1_sent_index, e2_sent_index]) + 1)
            min_event_sent, min_event_word = -1, -1
            max_event_sent, max_event_word = -1, -1
            if e1_sent_index < e2_sent_index:
                min_event_sent = e1_sent_index
                max_event_sent = e2_sent_index
                min_event_word = e1_data[1]
                max_event_word = e2_data[1]
            elif e1_sent_index > e2_sent_index:
                min_event_sent = e2_sent_index
                max_event_sent = e1_sent_index
                min_event_word = e2_data[1]
                max_event_word = e1_data[1]
            else:
                min_event_sent = e1_sent_index
                max_event_sent = e2_sent_index
                if e1_data[1] < e2_data[1]:
                    min_event_word = e1_data[1]
                    max_event_word = e2_data[1]
                else:
                    min_event_word = e2_data[1]
                    max_event_word = e1_data[1]
            total_text = []
            total_text_index = []
            for i in range(min_sent, max_sent + 1):
                tokens = doc_sents[i].split()
                total_text += tokens
                total_text_index += [idx + min_token_index + 1 + len(doc_sents) for idx, t in enumerate(tokens)]
                min_token_index += len(tokens)

            min_event_word_index = 0
            for i in range(min_sent, min_event_sent):
                min_event_word_index += len(doc_sents[i].split())
            min_event_word_index += min_event_word
            max_event_word_index = 0
            for i in range(min_sent, max_event_sent):
                max_event_word_index += len(doc_sents[i].split())
            max_event_word_index += max_event_word

            e1_word_index_for_llm = 0
            for i in range(min_sent, e1_sent_index):
                e1_word_index_for_llm += len(doc_sents[i].split())
            e1_word_index_for_llm += e1_data[1]
            e2_word_index_for_llm = 0
            for i in range(min_sent, e2_sent_index):
                e2_word_index_for_llm += len(doc_sents[i].split())
            e2_word_index_for_llm += e2_data[1]
            llm_payload = build_llm_payload(
                total_text,
                e1_word_index_for_llm,
                e2_word_index_for_llm,
                doc_id,
                (event_map[doc_id][e1], event_map[doc_id][e2])
            )
            indices = [
                [0, min_event_word_index],
                [min_event_word_index, min_event_word_index + 1],
                [min_event_word_index + 1, max_event_word_index],
                [max_event_word_index, max_event_word_index + 1],
                [max_event_word_index + 1, len(total_text)]
            ]
            example = [total_text, total_text_index, indices, rel_dict[label] if label != '' else label, doc_id,
                       event_map[doc_id][e1], event_map[doc_id][e2], syntax_graph, temporal_graph, llm_payload]
            if doc_id in test_files:
                test_data.append(example)
            elif doc_id in dev_files:
                dev_data.append(example)
            else:
                train_data.append(example)

    print('Loaded {} training examples'.format(len(train_data)))
    print('Loaded {} development examples'.format(len(dev_data)))
    print('Loaded {} test examples'.format(len(test_data)))


    def batch_examples(examples, split, batch_size):
        batched_examples = []
        for start in range(0, len(examples), batch_size):
            end = min(len(examples), start + batch_size)
            cur_examples = examples[start:end]
            example_texts = [x[0] for x in cur_examples]
            example_token_index = [x[1] for x in cur_examples]
            tokenized_inputs = tokenizer(
                example_texts,
                padding='max_length',
                truncation=True,
                is_split_into_words=True,
                max_length=512,
                return_tensors='pt'
            )
            renewed_indices = []
            doc_ids = []
            epairs = []
            syntax_graphs = []
            temporal_graphs = []
            cue_features = []
            llm_payloads = []
            for i, example in enumerate(cur_examples):
                doc_ids.append(example[4])
                epairs.append((example[5], example[6]))
                syntax_graphs.append(example[7])
                temporal_graphs.append(example[8])
                if split == 'test' and len(example) > 9:
                    llm_payloads.append(example[9])
                indices = example[2]
                cue_features.append(get_temporal_cue_features(example[0], indices))
                subtoken_indices = []
                start_index = 1
                mapping = tokenized_inputs.word_ids(batch_index=i)
                for cur_index_pair in indices:
                    subtoken_indices.append([start_index])
                    while mapping[start_index] != None and mapping[start_index] < cur_index_pair[1]:
                        start_index += 1
                    subtoken_indices[-1].append(start_index)
                renewed_indices.append(subtoken_indices)
            if split != 'test':
                tokenized_inputs['labels'] = torch.LongTensor([x[3] for x in cur_examples])
            batch = [tokenized_inputs, example_token_index, renewed_indices, doc_ids, epairs, syntax_graphs,
                     temporal_graphs, torch.FloatTensor(cue_features)]
            if split == 'test':
                batch.append(llm_payloads)
            batched_examples.append(batch)
        return batched_examples


    train_batches = batch_examples(train_data, 'train', args.batch_size)
    dev_batches = batch_examples(dev_data, 'dev', args.batch_size)
    test_batches = batch_examples(test_data, 'test', args.batch_size)

    context_encoder = BertTemporalOrdering(
        bert_encoder.cuda(),
        bert_config,
        use_evidence_attention=args.use_evidence_attention,
        evidence_boost=args.evidence_boost,
        middle_evidence_boost=args.middle_evidence_boost
    ).cuda()
    bert_output_units = 11 * bert_config.hidden_size
    syntax_gcn = RGCN_v1(bert_config.hidden_size, args.gcn_hidden_size, args.gcn_num_layers, len(syn_edge_type_dict),
                         args.gcn_type).cuda()
    temporal_gcn = RGCN_v1(bert_config.hidden_size, args.gcn_hidden_size, args.gcn_num_layers,
                           len(temporal_edge_type_dict), args.gcn_type).cuda()
    gcn_output_units = args.gcn_hidden_size * 2
    final_units = bert_output_units + gcn_output_units

    classifier = nn.Sequential(nn.Linear(final_units, len(rel_dict)), nn.ReLU(), nn.Softmax()).cuda()
    gcn_classifier = nn.Sequential(nn.Linear(gcn_output_units, len(rel_dict)), nn.ReLU(), nn.Softmax()).cuda()
    bert_classifier = nn.Sequential(nn.Linear(bert_output_units, len(rel_dict)), nn.ReLU(), nn.Softmax()).cuda()

    model = MulCo(
        syntax_gcn,
        temporal_gcn,
        context_encoder,
        classifier,
        args.dropout,
        args.k_hops,
        args.temp,
        bert_classifier,
        gcn_classifier,
        multi_scale=args.multi_scale,
        alpha_cl=args.alpha_cl,
        beta_tcl=args.beta_tcl,
        gamma_inv=args.gamma_inv,
        label_reverse_map=label_reverse_map,
        logic_loss_weight=args.logic_loss_weight,
        transitive_label_ids=transitive_label_ids,
        logic_conf_threshold=args.logic_conf_threshold
    )

    epochs = args.epochs
    trainer = BaseTrainer(args.learning_rate)
    trainer.train(model, train_batches, dev_batches, args.epochs, model_path, rel_dict, distance_flags,
                  args.accumulation_steps)

    if args.use_llm_refine:
        dev_confidence_records = None
        dev_labels_for_calibration = None
        if args.llm_auto_threshold_from_dev or getattr(args, 'llm_candidate_strategy', 'threshold') in ['risk_rank', 'defer_mlp'] or getattr(args, 'llm_risk_auto_target_count', False):
            print('Collecting dev predictions for LLM candidate calibration...')
            _, dev_confidence_records = trainer.test_with_confidence(model, dev_batches, model_path)
            dev_labels_for_calibration = _collect_labels_from_batches(dev_batches)
        if args.llm_auto_threshold_from_dev:
            print('Calibrating LLM candidate-selection threshold on dev set...')
            calibrate_llm_thresholds_from_dev(dev_confidence_records, dev_labels_for_calibration, args, model_path)
        if getattr(args, 'llm_candidate_strategy', 'threshold') in ['risk_rank', 'defer_mlp'] or getattr(args, 'llm_risk_auto_target_count', False):
            print('Calibrating dev risk profile for LLM risk-ranked candidate selection...')
            calibrate_llm_risk_selector_from_dev(dev_confidence_records, dev_labels_for_calibration, rel_dict, args, model_path)
        if getattr(args, 'llm_candidate_strategy', 'threshold') == 'defer_mlp':
            print('Training dev-based deferral MLP for candidate selection...')
            train_llm_deferral_mlp_from_dev(dev_confidence_records, dev_labels_for_calibration, rel_dict, args, model_path)
        if args.llm_refine_mode == 'correction_mlp':
            print('Training dev-based Top-2 Correction MLP for KEEP/SWITCH relation re-selection...')
            train_top2_correction_mlp_from_dev(dev_confidence_records, dev_labels_for_calibration, rel_dict, args, model_path)

        mulco_predictions, confidence_records = trainer.test_with_confidence(model, test_batches, model_path)
        base_results = compute_scores(mulco_predictions, gold_test_labels, rel_dict, distance_flags=distance_flags)
        print('Base MulCo results before LLM refinement:')
        print(base_results)
        json.dump(base_results, open(os.path.join(model_path, "eval_scores_before_llm.json"), "w"), indent=4)
        mulco_predictions, llm_stats = apply_llm_refinement(
            mulco_predictions, confidence_records, rel_dict, args, model_path
        )
    else:
        mulco_predictions = trainer.test(model, test_batches, model_path)

    results = compute_scores(mulco_predictions, gold_test_labels, rel_dict, distance_flags=distance_flags)
    print(results)

    json.dump(results, open(os.path.join(model_path, "eval_scores.json"), "w"), indent=4)
    json.dump(vars(args), open(os.path.join(model_path, "hyperparameters.json"), "w"), indent=4)
