from datasets import load_dataset

from random import randrange

def get_final_dataset(tokenizer):
    dataset_id = "knkarthick/dialogsum"
    dataset = load_dataset(dataset_id)

    import json

    with open("data/train_sick.json", "r", encoding="utf8") as f:
        train_sick = json.load(f)

    def update_dataset(example, index):
        example["dialogue"] = train_sick[index]
        return example

    updated_train = dataset["train"].map(update_dataset, with_indices=True)

    print(updated_train[0])

    with open("data/test_sick.json", "r", encoding="utf8") as f:
        test_sick = json.load(f)

    def update_dataset(example, index):
        example["dialogue"] = test_sick[index]
        return example

    updated_test = dataset["test"].map(update_dataset, with_indices=True)

    print(updated_test[0])


    from datasets import concatenate_datasets

    tokenized_inputs = concatenate_datasets([updated_train, updated_test]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

    tokenized_targets = concatenate_datasets([updated_train, updated_test]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

    def preprocess_function(sample,padding="max_length"): 

        # the instructions can be rewritten based on need

        # NOTE I tried the <I></I> and <sep></sep>, not much difference

        inputs = [f"Summarize the following dialogue with helping information after each utterance (between the special separator) and towards the topic of {sample['topic']}. " + item for item in sample["dialogue"]]

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
