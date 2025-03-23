# TextSummarization

This project implements an automated text summarization system using the T5-small model from Hugging Face’s transformers library.

**📄 Text Summarization** – Project Overview

This Text Summarization project utilizes T5 (Text-To-Text Transfer Transformer), a deep learning model, to generate concise and meaningful summaries from long-form text. It is designed to automate the summarization process, making lengthy content more digestible while preserving its essential meaning. The project is implemented using PyTorch and Hugging Face’s Transformers library to leverage a pre-trained T5 model.

**🌟 Features**

✔ Pre-Trained Transformer Model – Uses T5-small, a powerful NLP model trained for text generation.

✔ Text Cleaning & Formatting – Automatically corrects sentence structure before summarization.

✔ Customizable Summary Length – Allows tuning of maximum and minimum summary length.

✔ Batch Summarization Support – Can summarize multiple texts at once.

✔ Error Handling – Checks for empty inputs and invalid text formats.

✔ Tokenization & Encoding – Converts text into model-readable format using T5Tokenizer.

✔ Beam Search Optimization – Uses beam search decoding for high-quality text generation.

**🛠️ Technologies Used**

Python – Core programming language.

PyTorch – Deep learning framework for handling the T5 model.

Hugging Face Transformers – Provides pre-trained NLP models.

NLTK (Natural Language Toolkit) – Used for sentence and word tokenization.

T5 Model (T5ForConditionalGeneration) – A Transformer-based model optimized for text summarization.

**🚀 How It Works?**

1️⃣ Model Loading & Initialization

The T5-small model and T5Tokenizer are loaded using the load_model() function.

If the required packages (torch, transformers) are missing, they are installed automatically.

2️⃣ Text Preprocessing & Formatting

The input text is tokenized into sentences and words using nltk.tokenize.

The first word of each sentence is capitalized to maintain proper structure.

Unnecessary whitespace is removed.

3️⃣ Encoding the Input for the Model

The input text is prepended with "summarize: " to instruct the T5 model for summarization.

The text is tokenized and converted into input tensors using the T5Tokenizer.

It is truncated to 512 tokens (maximum limit for the model).

4️⃣ Generating the Summary

The model processes the encoded input and generates a summary using beam search decoding.

The length_penalty parameter ensures balanced summaries without excessive repetition.

The generated summary is decoded back into human-readable text.

The summary is formatted correctly before being returned.

5️⃣ Handling Batch Summarization

The function summarize_batch() processes multiple texts at once by applying summarize_text() to each input.

This enables summarization of multiple documents efficiently.

6️⃣ User Interaction

The script prompts the user to input text for summarization.

The input is cleaned, processed, and summarized.

The final generated summary is displayed.

**📂 Project Components**

Model Loading (load_model()) – Loads T5-small tokenizer and model.

Text Formatting (correct_text_format()) – Fixes capitalization and structure.

Text Summarization (summarize_text()) – Encodes input, generates summary, and decodes output.

Batch Processing (summarize_batch()) – Supports summarizing multiple texts.

User Input Handling – Prompts user for text and returns a summary.

**📌 Future Enhancements**

✅ Support for Larger T5 Models – Upgrade to T5-base or T5-large for improved performance.

✅ Fine-Tuning on Custom Datasets – Train the model on domain-specific content (legal, medical, finance).

✅ Web Interface Integration – Deploy using Flask or Streamlit for user-friendly interaction.

✅ Multi-Language Summarization – Extend support for multiple languages.

✅ Real-Time Summarization API – Create an API for summarizing text from external sources.

![Image](https://github.com/user-attachments/assets/b9c05be0-4feb-402d-923a-afa0cee47b5e)
