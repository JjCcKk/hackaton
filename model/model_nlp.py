from transformers import AutoTokenizer, AutoModelWithLMHead
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-sarcasm-twitter")

model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-sarcasm-twitter")

def eval_conversation(text):

  input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')
  output = model.generate(input_ids=input_ids, max_length=3)
  dec = [tokenizer.decode(ids) for ids in output]
  label = dec[0]

  return label


if __name__=="__main__":
    data = pd.read_json("/MUStARD/data/sarcasm_data.json")
    eval_conversation(" ".join(data[160]["context"]) + " " + data[160]["utterance"])