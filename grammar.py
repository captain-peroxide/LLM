import pandas as pd
import asyncio
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate

# Load the CSV file containing sentences
file_path = r'C:\Users\user\OneDrive\Desktop\llm\conversation_history.csv'
df = pd.read_csv(file_path)

# Remove NaN values in the specified column
column_name = df.columns[1]
df = df.dropna(subset=[column_name])

model_kwargs = {
    "mistral": {
        "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "model_file": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    }
}
llm_chains = {}
config = {"max_new_tokens": 256, "temperature": 0.3}

TEMPLATE = """<s>[INST] Please analyze the grammar of the sentence: '{sentence}' and provide feedback on its correctness.
[/INST] </s>
"""

async def analyze_grammar(contents: str, instance: pd.DataFrame):
    for model in model_kwargs:
        if model not in llm_chains:
            llm = CTransformers(**model_kwargs[model], config=config)
            prompt = PromptTemplate(template=TEMPLATE, input_variables=["sentence"])
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            llm_chains[model] = llm_chain

        response = await llm_chains[model].apredict(sentence=contents)
        
        # Check if "[/INST]" is in the response
        #
        # Check if Mistral indicates that the sentence is grammatically correct
        is_correct = "This sentence is grammatically correct." in response
        
        # Count the number of grammar mistakes indicated by Mistral
        mistakes_count = response.count("This sentence contains")
        
        # Update the DataFrame with the analysis results
        instance.loc[instance[column_name] == contents, 'Mistakes Count'] = mistakes_count
        instance.loc[instance[column_name] == contents, 'Is Correct'] = is_correct
async def analyze_all_sentences(df):
    # Apply the grammar analysis function to each sentence in the DataFrame
    for _, row in df.iterrows():
        await analyze_grammar(row[column_name], df)

# Run the async function
asyncio.run(analyze_all_sentences(df))

# Display the DataFrame with the analysis results
output_file_path = r'C:\Users\user\OneDrive\Desktop\llm\analysis_results.csv'
df.to_csv(output_file_path, index=False)
print(f"Analysis results saved to {output_file_path}")