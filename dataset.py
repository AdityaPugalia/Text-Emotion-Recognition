import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAPPING = {0 : 'sadness', 1 : 'joy', 2 : 'love', 3 : 'anger', 4 : 'fear', 5 : 'surprise'}

emotion_df = pd.read_csv('data/emotions.csv')
print("Original size:", len(emotion_df))
emotion_df = emotion_df.drop_duplicates(subset="text", keep='first').reset_index(drop=True)
print("After deduplication:", len(emotion_df))
emotion_df_train, emotion_df_test = train_test_split(emotion_df, test_size=0.2, random_state=42)

emotion_df_train, emotion_df_val = train_test_split(emotion_df_train, test_size=0.2, random_state=42)

emotion_df_train.to_csv('data/emotion_train.csv', index=False)
emotion_df_val.to_csv('data/emotion_val.csv', index=False)
emotion_df_test.to_csv('data/emotion_test.csv', index=False)

