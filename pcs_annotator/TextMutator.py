from .LLM import LLM
from .Tools import *

class TextMutator(LLM):
    def __init__(self, model_name="llama-3.1-8b-instant", token=None):
        super().__init__(model_name, token)
        self.mr_templates = {
            "passive_active": self._passive_active_prompt,
            "double_negation": self._double_negation_prompt,
            "synonym": self._synonym_prompt
        }

    def register_mr(self, mr_name, template_fn):
        """Allows users to register a new MR and its template-generating function."""
        self.mr_templates[mr_name] = template_fn

    def MutateText(self, text, mr):
        if mr not in self.mr_templates:
            raise ValueError(f"Unknown MR: {mr}. Please register it using `register_mr()`.")

        prompt = self.mr_templates[mr](text)

        try:
            ans = self.llm.invoke(prompt)
            for key in ["new_text:", "New Text:", "New_Text:"]:
                if key in ans:
                    ans = ans.split(key)[1]
                    break
            return trim_text(ans)
        except Exception as e:
            print(f"Error querying model: {e}")
            return None

    def _passive_active_prompt(self, text):
        return f"""Review the text below and transform all sentences by converting active voice to passive voice and vice versa, where appropriate.
            Provide only the revised text, without extra information, starting with the term "new_text:"
            Original Text:
            {text}
            """

    def _double_negation_prompt(self, text):
        return f"""Review the text below and transform affirmative sentences into double negation sentences.
            Provide only the revised text, without extra information, starting with the term "new_text:"
            Original Text:
            {text}
            """

    def _synonym_prompt(self, text):
        return f"""Review the text below and replace key words with their synonyms. Ensure that the transformed sentences retain equivalent meanings.
            Provide only the revised text, without extra information, starting with the term "new_text:"
            Original Text:
            {text}
            """
