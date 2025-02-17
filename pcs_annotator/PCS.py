from TextMutator import TextMutator
from Annotator import Annotator
import pandas as pd
import os
import numpy as np
from scipy.optimize import differential_evolution
from tqdm import tqdm



class PCS:
    def __init__(self, prompt, dataset_path, annotators=["llama3-8b-8192", "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it"]):
        self.prompt=prompt
        if os.path.exists(dataset_path):
            self.dataset_path = dataset_path
        else:
            raise FileNotFoundError(f"Invalid dataset path {dataset_path}") 
        self.annotator_model_names = annotators
        self.annotators = []
        self.textMutator = TextMutator()

        self.annotations_filename = self.dataset_path.rsplit(".csv", 1)[0] + "_annotations.csv"
        self.dataset = pd.read_csv(dataset_path)
        self.possible_labels = self.dataset["label"].unique().tolist()
        for annotator_model_name in self.annotator_model_names:
            self.annotators.append(Annotator(prompt=self.prompt, labels=self.possible_labels, model_name=annotator_model_name))
        self.optimal_llm_weights = [1] * len(self.annotators)
        self.optimal_mr_weights = [1] * 4
        self.optimal_label_thresholds = [0.5] * len(self.possible_labels)
        
        print("Labelling the dataset...")
        self.create_annotations()
        print("Generated the dataset annotations.")

        print("Optimizing weights...")
        self.PDE()
        print("Done Optimizing weights.")

    def create_annotations(self):
        csv_filename = self.annotations_filename

        if os.path.exists(csv_filename):
            # Read the existing dataset
            existing_df = pd.read_csv(csv_filename)
            if not existing_df.empty:
                last_index = existing_df.index[-1] + 1  # Get last index and continue from there
            else:
                last_index = 0
        else:
            columns = ["label", "text", "text_mr1", "text_mr2", "text_mr3"]
            for annotator in self.annotators:
                columns.append(annotator.model_name+"_text")
                columns.append(annotator.model_name+"_text_mr1")
                columns.append(annotator.model_name+"_text_mr2")
                columns.append(annotator.model_name+"_text_mr3")

            existing_df = pd.DataFrame(columns=columns)
            existing_df.to_csv(csv_filename, index=False)
            last_index = 0

        with tqdm(total=len(self.dataset), initial=last_index, desc="Processing Annotations", unit="sample") as pbar:

            for index in range(last_index, len(self.dataset)):
                row = self.dataset.iloc[index]
                label = row['label']
                text = row['text']
                text_mr1 = self.textMutator.MutateText(text=text, mr="passive_active")
                text_mr2 = self.textMutator.MutateText(text=text, mr="double_negation")
                text_mr3 = self.textMutator.MutateText(text=text, mr="synonym")

                existing_df.at[index, "label"] = label
                existing_df.at[index, "text"] = text
                existing_df.at[index, "text_mr1"] = text_mr1
                existing_df.at[index, "text_mr2"] = text_mr2
                existing_df.at[index, "text_mr3"] = text_mr3

                for annotator in self.annotators:
                    existing_df.at[index, annotator.model_name+"_text"] = annotator.annotate(text)[0]
                    existing_df.at[index, annotator.model_name+"_text_mr1"] = annotator.annotate(text_mr1)[0]
                    existing_df.at[index, annotator.model_name+"_text_mr2"] = annotator.annotate(text_mr2)[0]
                    existing_df.at[index, annotator.model_name+"_text_mr3"] = annotator.annotate(text_mr3)[0]

                row_df = existing_df.iloc[[index]]
                row_df.to_csv(csv_filename, mode='a', index=False, header=not os.path.exists(csv_filename))
                    
                pbar.update(1)  # Update progress bar

    
    def get_majority_vote_label(self, majority_votes, possible_labels=None):
        if not possible_labels:
            possible_labels = self.possible_labels

        vote_counts = {label.strip().lower(): 0 for label in possible_labels}

        for vote in majority_votes:
            if not pd.isna(vote):
                lowered_vote = vote.strip().lower()
                if lowered_vote in vote_counts:
                    vote_counts[lowered_vote] += 1

        max_votes = max(vote_counts.values()) 
                
        for label, count in vote_counts.items():
            if count == max_votes:
                return label

    def get_pcs_label(self, llm_labels, w_llms=None, w_mrs=None, label_thresholds=None, possible_labels=None):
        if possible_labels is None:
            possible_labels = self.possible_labels
        if w_llms is None:
            w_llms = self.optimal_llm_weights
        if w_mrs is None:
            w_mrs = self.optimal_mr_weights
        if label_thresholds is None:
            label_thresholds = self.optimal_label_thresholds

        num_labels = len(possible_labels)
        pcs_labels = [0] * num_labels 
        
        for i in range(len(w_llms)):
            pcs_label_llm = [0] * num_labels
            for j in range(len(llm_labels[i])): #Iterate through MR labels 
                for k in range(len(pcs_label_llm)):
                    if not pd.isna(llm_labels[i][j]):
                        if llm_labels[i][j].strip().lower() == possible_labels[k].strip().lower():
                            pcs_label_llm[k] += w_mrs[j]

            for j in range(len(pcs_label_llm)):
                # pcs_label_llm[j] = pcs_label_llm[j]/sum(w_mrs)
                if sum(w_mrs) != 0:
                    pcs_label_llm[j] = pcs_label_llm[j] / sum(w_mrs)
                else:
                    pcs_label_llm[j] = 0 
                
            for j in range(len(pcs_label_llm)):
                pcs_labels[j] += w_llms[i] * pcs_label_llm[j]

        for i in range(num_labels):
            if sum(w_llms) != 0:
                pcs_labels[i] = pcs_labels[i]/sum(w_llms)
            else:
                pcs_labels[i] = 0 

        for i in range(num_labels):
            if pcs_labels[i] > label_thresholds[i]:
                final_pred = possible_labels[i]
                return final_pred

        majority_votes = []
        for i in range(len(w_llms)):
            majority_votes.append(llm_labels[i][0])
    
        return self.get_majority_vote_label(majority_votes, possible_labels)


    def Objective(self, params, llm_preds, NUM_LLMS, NUM_MRS, manual_labels):
        w_mrs = params[:NUM_MRS]  # Weights for 4 MRs
        w_llms = params[NUM_MRS:NUM_MRS+NUM_LLMS]  # Weights for 3 LLMs
        label_thresholds = params[NUM_MRS+NUM_LLMS:NUM_MRS+NUM_LLMS+len(self.possible_labels)]
        predictions = []

        predictions = [
            self.get_pcs_label(llm_preds[i, :], w_llms, w_mrs, label_thresholds)
            for i in range(len(manual_labels))
        ]
            
        # Calculate accuracy
        accuracy = np.mean([
            manual_labels[i].strip().lower() == predictions[i].strip().lower()
            for i in range(len(manual_labels))
        ])
        return -accuracy  # We want to maximize accuracy, so we minimize negative accuracy

    def PDE(self):
        df = pd.read_csv(self.annotations_filename)

        # Define model column mappings
        model_columns = []
        for annotator in self.annotators:
            model_columns.append(df[[f"{annotator.model_name}_text", f"{annotator.model_name}_text_mr1", f"{annotator.model_name}_text_mr2", f"{annotator.model_name}_text_mr3"]])
        llm_preds = np.stack(model_columns, axis=1)

        manual_labels = df['label'].values
        size, NUM_LLMS, NUM_MRS = llm_preds.shape[:3]

        bounds = [(0, 1)] * (NUM_LLMS + NUM_MRS + len(self.possible_labels))
        
        result = differential_evolution(self.Objective, bounds, args=(llm_preds, NUM_LLMS, NUM_MRS, manual_labels))
        
        optimal_params = result.x
        
        self.optimal_mr_weights = optimal_params[:NUM_MRS]
        self.optimal_llm_weights = optimal_params[NUM_MRS:NUM_MRS+NUM_LLMS]
        self.optimal_label_thresholds = optimal_params[NUM_MRS+NUM_LLMS:NUM_MRS+NUM_LLMS+len(self.possible_labels)]
        self.Accuracy = -result.fun
        
    def annotate(self, text):
        text_mr1 = self.textMutator.MutateText(text=text, mr="passive_active")
        text_mr2 = self.textMutator.MutateText(text=text, mr="double_negation")
        text_mr3 = self.textMutator.MutateText(text=text, mr="synonym")
        llm_preds = []
        for annotator in self.annotators:
            annotations = []
            annotations.append(annotator.annotate(text)[0])
            annotations.append(annotator.annotate(text_mr1)[0])
            annotations.append(annotator.annotate(text_mr2)[0])
            annotations.append(annotator.annotate(text_mr3)[0])
            llm_preds.append(annotations)

        pcs_label = self.get_pcs_label(llm_preds)
        return pcs_label

