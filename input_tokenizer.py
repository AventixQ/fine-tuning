def tokenize_function(input_data, tokenizer):
    inputs = input_data["input"]
    outputs = input_data["output"]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length").input_ids
    
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
    
    model_inputs["labels"] = labels
    return model_inputs