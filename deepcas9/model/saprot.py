import esm
import time
import itertools
import torch
import torch.nn as nn
from torch import device

from esm.model.esm2 import ESM2


foldseek_seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"


class SaProt(nn.Module):
    def __init__(self, model: nn.Module, alphabet: object):
        super(SaProt, self).__init__()
        self.model, self.alphabet = model, alphabet
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

    def saprot_inference(self, saprot_tokens, device='cuda'):
        """
        saprot tokens:
            Has already been processed by the batch converter
        """
        saprot_batch_tokens = saprot_tokens
        batch_lens = (saprot_batch_tokens != self.alphabet.padding_idx).sum(1)
        saprot_batch_tokens = saprot_batch_tokens.to(device)
        results = self.model(saprot_batch_tokens, repr_layers=[12], return_contacts=True)
        token_representations = results["representations"][12]
        sequence_representations = token_representations[:, 1: batch_lens[0] - 1, :]
        return sequence_representations


def load_esm_saprot(path: str):
    """
    Load SaProt model of esm version.
    Args:
        path: path to SaProt model

    Source from SaProt: https://github.com/westlake-repl/SaProt/blob/main/utils/esm_loader.py
    """

    # Initialize the alphabet
    tokens = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
    for seq_token, struc_token in itertools.product(foldseek_seq_vocab, foldseek_struc_vocab):
        token = seq_token + struc_token
        tokens.append(token)

    alphabet = esm.data.Alphabet(standard_toks=tokens,
                                 prepend_toks=[],
                                 append_toks=[],
                                 prepend_bos=True,
                                 append_eos=True,
                                 use_msa=False)

    alphabet.all_toks = alphabet.all_toks[:-2]
    alphabet.unique_no_split_tokens = alphabet.all_toks
    alphabet.tok_to_idx = {tok: i for i, tok in enumerate(alphabet.all_toks)}

    # Load weights
    data = torch.load(path)
    weights = data["model"]
    config = data["config"]

    # Initialize the model
    model = ESM2(
        num_layers=config["num_layers"],
        embed_dim=config["embed_dim"],
        attention_heads=config["attention_heads"],
        alphabet=alphabet,
        token_dropout=config["token_dropout"],
    )

    load_weights(model, weights)
    return model, alphabet


def load_weights(model, weights):
    """
    Source from SaProt: https://github.com/westlake-repl/SaProt/blob/main/utils/esm_loader.py
    """
    model_dict = model.state_dict()

    unused_params = []
    missed_params = list(model_dict.keys())

    for k, v in weights.items():
        if k in model_dict.keys():
            model_dict[k] = v
            missed_params.remove(k)

        else:
            unused_params.append(k)

    if len(missed_params) > 0:
        print(f"\033[31mSome weights of {type(model).__name__} were not "
              f"initialized from the model checkpoint: {missed_params}\033[0m")

    if len(unused_params) > 0:
        print(f"\033[31mSome weights of the model checkpoint were not used: {unused_params}\033[0m")

    model.load_state_dict(model_dict)
