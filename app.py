import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

# Load and cache the GPT-2 model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Fine-tune the model (correct labels added)
@st.cache_resource
def fine_tune_model(_model, _tokenizer):
    dataset = load_dataset("text", data_files={"train": "random_story.txt"})

    def tokenize_function(examples):
        inputs = _tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = inputs["input_ids"].copy()  # Add labels for language modeling
        return inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10,
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=5
    )

    trainer = Trainer(
        model=_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )

    trainer.train()
    return _model

# Fine-tune the model
model = fine_tune_model(model, tokenizer)

# Streamlit UI
st.title("GPT-2 Story Generator")
prompt = st.text_area("Enter your story prompt:", "In an enchanted forest, a mysterious traveler")

if st.button("Generate"):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, do_sample=True, temperature=0.8)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.markdown("Generated Story")
    st.markdown(generated_text)
