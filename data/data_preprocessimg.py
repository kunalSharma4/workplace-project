
import pandas as pd 
#Importing data


# Display the first few rows of the dataset
#print(emails.head())  

#removing unnamed column

def clean_emails(df):
     if 'Unnamed: 0' in df.columns:
          df = df.drop(columns=['Unnamed: 0'])
     return df


#removing empty spaces

def remove_bad_rows(df):
     df = df[df['Email Text'].str.lower().str.strip() != 'empty']

        # Keep only rows where 'Email Text' is a valid string
     df = df[df['Email Text'].apply(lambda x: isinstance(x, str))]

     return df

def main():
    emails = pd.read_csv("data/Phishing_Email.csv")
    emails = clean_emails(emails)
    emails = remove_bad_rows(emails)
    emails.to_csv("Cleaned_data.csv", index=False)
    print(emails.head())
if __name__ == "__main__":
        main()


