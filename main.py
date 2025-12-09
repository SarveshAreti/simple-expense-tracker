from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")
ner_model = pipeline("ner", aggregation_strategy="simple")
summarizer = pipeline("summarization")

text = input("Enter text: ")

print("\n--- Sentiment ---")
print(sentiment_model(text))

print("\n--- Named Entities ---")
print(ner_model(text))

print("\n--- Summary ---")
print(summarizer(text, max_length=50, min_length=10)[0]['summary_text'])
