Since the pLM used in this project includes ESM-2 and SaProt, we need to modify the source code in the `fair-esm` package.

**Modify the `.conda/envs/protein/lib/python3.10/site-packages/esm/data.py` file**.
1. Import necessary python packages:
    ```python
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    import os
    ```
2. Replace the `get_batch_converter` function (line 139 - 143) with the following code:
    ```python
        def get_batch_converter(self, truncation_seq_length: int = None, model_name: str = None):
        if self.use_msa:
            return MSABatchConverter(self, truncation_seq_length)
        else:
            return BatchConverter(self, truncation_seq_length, model_name)
    ```
3. Replace the `BatchConverter` class (line 263 - 265) with the following code:
    ```python
        def __init__(self, alphabet, truncation_seq_length: int = None, model_name: str = None):
            self.alphabet = alphabet
            self.truncation_seq_length = truncation_seq_length
            self.model_name = model_name

        def encode_sequence(self, seq_str):
            return self.alphabet.encode(seq_str)
    ```
4. Replace the `seq_encoded_list` (line 271) with the following code:
    ```python
        if self.model_name == 'esm':
            seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        elif self.model_name == 'saprot':
            with ProcessPoolExecutor(max_workers=os.cpu_count()//2) as executor:
                seq_encoded_list = list(tqdm(executor.map(self.encode_sequence, seq_str_list), total=len(seq_str_list)))
    ```

Since preprocessing the protein sequence for SaProt tokenizer is time-consuming, we make the above modifications.
