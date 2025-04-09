import esm
import time
import torch.nn as nn
import torch

class ESM2(nn.Module):
    def __init__(self, esm_file):
        super(ESM2, self).__init__()
        self.esm_file = esm_file
        self.model, self.alphabet = self._load_esm()
        self.batch_converter = self.alphabet.get_batch_converter(model_name='esm')
        self.model.eval()
        self.model = self.model.to('cuda')

    def unfreeze_layers(self, layer_idxs: list):
        """
        Finetune the model with the specified layers
        """
        for param in self.model.parameters():
            param.requires_grad = False
        for layer_idx in layer_idxs:
            for param in self.model.layers[layer_idx].parameters():
                param.requires_grad = True

    def esm_inference(self, seq_list, device='cuda'):
        """
        seq_list should be in the format of [(pdb_id, sequence), ...]
        For example:
        pdb_ids = ['PDB1', 'PDB2']
        seqs = ['MKTLLVLLYAFVIGWLTSTS', 'MKTLLVLLYAFVIGWLTSTS']
        seq_list = list(zip(pdb_ids, seqs))
        """
        _, _, esm2_batch_tokens = self.batch_converter(seq_list)
        batch_lens = (esm2_batch_tokens != self.alphabet.padding_idx).sum(1)
        esm2_batch_tokens = esm2_batch_tokens.to(device)
        results = self.model(esm2_batch_tokens, repr_layers=[12], return_contacts=True)
        token_representations = results["representations"][12]
        sequence_representations = token_representations[:, 1: batch_lens[0] - 1]

        return sequence_representations

    def esm_inference_gradient(self, seq_list, device='cuda', require_grad=False):
        """
        seq_list should be in the format of [(pdb_id, sequence), ...]
        For example:
        pdb_ids = ['PDB1', 'PDB2']
        seqs = ['MKTLLVLLYAFVIGWLTSTS', 'MKTLLVLLYAFVIGWLTSTS']
        seq_list = list(zip(pdb_ids, seqs))
        """
        _, _, esm2_batch_tokens = self.batch_converter(seq_list)
        batch_lens = (esm2_batch_tokens != self.alphabet.padding_idx).sum(1)
        esm2_batch_tokens = esm2_batch_tokens.to(device)
        if require_grad:
            need_head_weights, padding_mask, x = (
                self._esm2_embed_stage_1(tokens=esm2_batch_tokens, need_head_weights=False, return_contacts=False))
            results = self._esm2_embed_stage_2(tokens=esm2_batch_tokens, repr_layers=[12], need_head_weights=False,
                                               return_contacts=False, padding_mask=padding_mask, x=x)
        else:
            results = self.model(esm2_batch_tokens, repr_layers=[12], return_contacts=True)
        token_representations = results["representations"][12]
        sequence_representations = token_representations[:, 1: batch_lens[0] - 1]
        return sequence_representations

    def _esm2_embed_stage_1(self, tokens, need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.model.padding_idx)  # B, T

        x = self.model.embed_scale * self.model.embed_tokens(tokens)
        x.requires_grad = True
        return need_head_weights, padding_mask, x

    def _esm2_embed_stage_2(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False,
                            padding_mask=None, x=None):
        if self.model.token_dropout:
            x.masked_fill_((tokens == self.model.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.model.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.model.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.model.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.model.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.model.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def _load_esm(self):
        start_time = time.time()
        model, alphabet = esm.pretrained.load_model_and_alphabet(self.esm_file)
        end_time = time.time()
        print("Load ESM2 model time: ", end_time - start_time, "s")
        return model, alphabet
