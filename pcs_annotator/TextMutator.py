from .LLM import LLM
from .Tools import *

class TextMutator(LLM):
    def __init__(self, model_name="llama-3.1-8b-instant", token=None):
        super().__init__(model_name, token)


    def MutateText(self, text, mr):
        if mr == "passive_active":
            prompt = f"""Review the text below and transform all sentences by converting active voice to passive voice and vice versa, where appropriate.

            Provide only the revised text, without extra information, starting with the term "new_text:"

            Original Text:
            {text}
            """
            
        if mr == "double_negation":
            prompt = f"""Review the text below and transform affirmative sentences into double negation sentences. (Double negation means using two negative elements within a clause or sentence, typically leading to a positive implication.) Ensure that the transformations maintain equivalent meanings.

            Provide only the revised text, without extra information, starting with the term "new_text:"

            Original Text:
            {text}
            """
        

        if mr == "synonym":
            prompt = f"""Review the text below and Replace key words with their synonyms. Ensure that the transformed sentences retain equivalent meanings.

            Provide only the revised text, without extra information, starting with the term "new_text:"

            Original Text:
            {text}
            """

        try:
            ans = self.llm.invoke(prompt)
            if "new_text:" in ans:
                ans = ans.split("new_text:")[1]
            elif "New Text:" in ans:
                ans = ans.split("New Text:")[1]
            elif "New_Text:" in ans:
                ans = ans.split("New_Text:")[1]

            ans = trim_text(ans)

            return ans        
        
        except Exception as e:
            
            print(f"Error querying model: {e}")
            return None

