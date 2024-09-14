import nanogcg
import torch
import json
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

model_id = "E:\sona\lmdeploy\internlm2-chat-1_8b"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)

data = pd.read_csv('dataset\harmful_behaviors.csv')


for i in range(50):
    for fix_seed in ([33, 66, 99]):

        message = data["goal"][i]
        target =  data["target"][i]

        config = GCGConfig(
            search_width=200,
            seed=fix_seed,
            verbosity="INFO",
            early_stop=True,
        )

        result = nanogcg.run(model, tokenizer, message, target, config)
        print(result)