import requests
from bs4 import BeautifulSoup
import csv

def extract_questions_and_answers(url, num_questions):
    data = []
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        for i in range(1, num_questions + 1):
            question_id = f"Q{i}"
            question_tag = soup.find('a', attrs={'name': question_id})
            if question_tag:
                question_text = question_tag.find_next('span').text.strip()
                answer_tag = question_tag.find_next('p')
                if answer_tag:
                    answer_text = answer_tag.text.strip()
                else:
                    answer_text = "Answer not found."
                data.append((question_text, answer_text))
            else:
                print(f"Question {i} not found.")
    else:
        print("Failed to retrieve the webpage.")
    return data

def save_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Answer'])
        writer.writerows(data)

# Example usage:
url = "https://www.geeksforgeeks.org/data-science-interview-questions-and-answers/"
num_questions = 100
data = extract_questions_and_answers(url, num_questions)
csv_filename = "questions_and_answers.csv"
save_to_csv(data, csv_filename)
print(f"Data saved to {csv_filename} successfully.")
