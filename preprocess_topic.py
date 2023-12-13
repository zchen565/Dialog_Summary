from datasets import load_dataset
from datasets import concatenate_datasets
from random import randrange

def get_final_dataset(tokenizer):
    dataset_id = "knkarthick/dialogsum"
    dataset = load_dataset(dataset_id)

    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    def preprocess_function(sample,padding="max_length"): 

        # the instructions can be rewritten based on need

        # NOTE I tried the <I></I> and <sep></sep>, not much difference

        inputs = [f"Summarize the following dialogue towards the topic of {sample['topic']}. " + item for item in sample["dialogue"]]

        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])

    return tokenized_dataset
