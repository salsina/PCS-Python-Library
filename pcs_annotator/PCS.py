from .TextMutator import TextMutator
from .Annotator import Annotator
import pandas as pd
import os
import numpy as np
from scipy.optimize import differential_evolution
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


class PCS:
    def __init__(self, prompt, dataset_path, annotators=["llama3-8b-8192", "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it"], textmutator="llama-3.1-8b-instant", GROQ_API_KEY=None, OPENAI_API_KEY=None, ANTHROPIC_API_KEY=None, HUGGINGFACE_API_KEY=None, train=True):
        self.prompt=prompt
        if os.path.exists(dataset_path):
            self.dataset_path = dataset_path
        else:
            raise FileNotFoundError(f"Invalid dataset path {dataset_path}") 
        self.annotator_model_names = annotators
        self.annotators = []

        if textmutator.startswith("llama") or textmutator.startswith("gemma"):
            self.textMutator = TextMutator(model_name=textmutator, token=GROQ_API_KEY)
        elif textmutator.startswith("gpt"):
            self.textMutator = TextMutator(model_name=textmutator, token=OPENAI_API_KEY)
        elif textmutator.startswith("claude"):
            self.textMutator = TextMutator(model_name=textmutator, token=ANTHROPIC_API_KEY)
        elif textmutator.startswith("mistral"):
            self.textMutator = TextMutator(model_name=textmutator, token=HUGGINGFACE_API_KEY)
        else:
            raise ValueError(f"Unsupported text mutator model name: {textmutator}")


        self.annotations_filename = self.dataset_path.rsplit(".csv", 1)[0] + "_annotations.csv"
        self.dataset = pd.read_csv(dataset_path)
        self.possible_labels = self.dataset["label"].unique().tolist()
        for annotator_model_name in self.annotator_model_names:
            if annotator_model_name.startswith("llama") or annotator_model_name.startswith("gemma"):
                token = GROQ_API_KEY
            elif annotator_model_name.startswith("gpt"):
                token = OPENAI_API_KEY
            elif annotator_model_name.startswith("claude"):
                token = ANTHROPIC_API_KEY
            elif annotator_model_name.startswith("mistral") :
                token = HUGGINGFACE_API_KEY
            else:
                raise ValueError(f"Unsupported annotator model name: {annotator_model_name}")

            self.annotators.append(Annotator(prompt=self.prompt, labels=self.possible_labels, model_name=annotator_model_name, token=token))
        self.optimal_llm_weights = [1] * len(self.annotators)
        self.optimal_mr_weights = [1] * 4
        # self.optimal_label_thresholds = [0.5] * len(self.possible_labels)
        
        if train:
            print("Labelling the dataset...")
            self.create_annotations()
            print("Generated the dataset annotations.")

        print("Optimizing weights...")
        # self.PDE()
        self.LR()
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



    def get_pcs_label(self, llm_labels, w_llms=None, w_mrs=None, possible_labels=None):
        if possible_labels is None:
            possible_labels = self.possible_labels
        if w_llms is None:
            w_llms = self.optimal_llm_weights
        if w_mrs is None:
            w_mrs = self.optimal_mr_weights

        num_labels = len(possible_labels)
        pcs_labels = np.zeros(num_labels)

        for i in range(len(w_llms)):
            pcs_label_llm = np.zeros(num_labels)
            for j in range(len(llm_labels[i])): #Iterate through MR labels 
                for k in range(len(pcs_label_llm)):
                    if llm_labels[i][j]:
                        if llm_labels[i][j].strip().lower() == possible_labels[k].strip().lower():
                            pcs_label_llm[k] += w_mrs[j]

            # Normalize by sum of MR weights

            if sum(w_mrs) != 0:
                pcs_label_llm = pcs_label_llm / sum(w_mrs)
                
            # Weight by LLM weights and accumulate
            pcs_labels += w_llms[i] * pcs_label_llm

        # Normalize by sum of LLM weights

        if sum(w_llms) != 0:
            pcs_labels = pcs_labels / sum(w_llms)

        return pcs_labels

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
        if os.path.exists(self.annotations_filename):
            df = pd.read_csv(self.annotations_filename)
        else:
            return
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


    def compute_features(self, llm_preds, selected_indices, optimize_params=["LLM", "MR"]):
        """
        Compute feature matrix for linear regression.
        Each row represents an example, and columns represent feature combinations based on optimize_params.
        
        Args:
            llm_preds: Predictions from LLMs (shape: [n_samples, n_llms, n_mrs])
            labels: List of possible label categories
            selected_indices: Indices of samples to use
            optimize_params: List of parameters to optimize (can include "LLM", "MR")
        """
        num_examples = len(selected_indices)
        num_llms = llm_preds.shape[1]
        num_mrs = llm_preds.shape[2]
        num_labels = len(self.possible_labels)
        
        # Determine feature structure based on optimize_params
        if "LLM" in optimize_params and "MR" in optimize_params:
            # Original case: optimize both LLM and MR
            feature_dim = num_llms * num_mrs * num_labels
            
            # Initialize feature matrix
            X = np.zeros((num_examples, feature_dim))
            
            for idx, sample_idx in enumerate(selected_indices):
                for i in range(num_llms):
                    for j in range(num_mrs):
                        for k, label in enumerate(self.possible_labels):
                            # Check if this LLM-MR combination predicts this label
                            if llm_preds[sample_idx, i, j].strip().lower() == label.strip().lower():
                                feature_idx = i * (num_mrs * num_labels) + j * num_labels + k
                                X[idx, feature_idx] = 1
                                
        elif "LLM" in optimize_params:
            # Only optimize LLM weights
            feature_dim = num_llms * num_labels
            
            # Initialize feature matrix
            X = np.zeros((num_examples, feature_dim))
            
            for idx, sample_idx in enumerate(selected_indices):
                for i in range(num_llms):
                    # Aggregate across all MRs for each LLM
                    llm_preds_aggregated = []
                    for j in range(num_mrs):
                        llm_preds_aggregated.append(llm_preds[sample_idx, i, j].strip().lower())
                    
                    for k, label in enumerate(self.possible_labels):
                        # Count how many times this LLM predicts this label across MRs
                        count = llm_preds_aggregated.count(label.strip().lower())
                        if count > 0:
                            feature_idx = i * num_labels + k
                            X[idx, feature_idx] = count / num_mrs  # Normalize by number of MRs
                            
        elif "MR" in optimize_params:
            # Only optimize MR weights
            feature_dim = num_mrs * num_labels
            
            # Initialize feature matrix
            X = np.zeros((num_examples, feature_dim))
            
            for idx, sample_idx in enumerate(selected_indices):
                for j in range(num_mrs):
                    # Aggregate across all LLMs for each MR
                    mr_preds_aggregated = []
                    for i in range(num_llms):
                        mr_preds_aggregated.append(llm_preds[sample_idx, i, j].strip().lower())
                    
                    for k, label in enumerate(self.possible_labels):
                        # Count how many times this MR predicts this label across LLMs
                        count = mr_preds_aggregated.count(label.strip().lower())
                        if count > 0:
                            feature_idx = j * num_labels + k
                            X[idx, feature_idx] = count / num_llms  # Normalize by number of LLMs
        else:
            # If optimize_params is empty or contains invalid parameters, default to both
            return self.compute_features(llm_preds, selected_indices, ["LLM", "MR"])
        
        return X

    def LR(self):
        if os.path.exists(self.annotations_filename):
            df = pd.read_csv(self.annotations_filename)
        else:
            return
        # Define model column mappings
        model_columns = []
        for annotator in self.annotators:
            model_columns.append(df[[f"{annotator.model_name}_text", f"{annotator.model_name}_text_mr1", f"{annotator.model_name}_text_mr2", f"{annotator.model_name}_text_mr3"]])
        llm_preds = np.stack(model_columns, axis=1)

        manual_labels = df['label'].values
        size, NUM_LLMS, NUM_MRS = llm_preds.shape[:3]

        NUM_LABELS = len(self.possible_labels)
        optimize_params = ["LLM", "MR"]
        N_samples = llm_preds.shape[0]
        selected_indices = np.random.choice(N_samples, size, replace=False)
        
        # Prepare target variable (one-hot encoded labels)
        y = np.zeros((len(selected_indices), NUM_LABELS))
        for i, idx in enumerate(selected_indices):
            label_idx = self.possible_labels.index(manual_labels[idx].strip().lower())
            y[i, label_idx] = 1
        
        # Compute features based on optimize_params
        X = self.compute_features(llm_preds, selected_indices, optimize_params)

        # Train linear regression model
        model = LinearRegression(positive=True)  # Ensure positive weights
        model.fit(X, y)
        
        # Extract weights from coefficients
        coefficients = model.coef_
        weights = np.sum(coefficients, axis=0)
        # Initialize weights
        optimal_llm_weights = None
        optimal_mr_weights = None
        
        # Extract and reshape weights based on optimize_params
        if "LLM" in optimize_params and "MR" in optimize_params:
            # Original case: optimize both LLM and MR
            reshaped_weights = weights.reshape(NUM_LLMS, NUM_MRS, NUM_LABELS)
            # Average across labels to get LLM and MR weights
            optimal_llm_weights = np.mean(np.sum(reshaped_weights, axis=1), axis=1)
            optimal_mr_weights = np.mean(np.sum(reshaped_weights, axis=0), axis=1)
        elif "LLM" in optimize_params:
            # Only optimize LLM weights
            reshaped_weights = weights.reshape(NUM_LLMS, NUM_LABELS)
            optimal_llm_weights = np.mean(reshaped_weights, axis=1)
            
            # Use uniform weights for MRs
            optimal_mr_weights = np.ones(NUM_MRS) / NUM_MRS
            
        elif "MR" in optimize_params:
            # Only optimize MR weights
            reshaped_weights = weights.reshape(NUM_MRS, NUM_LABELS)
            optimal_mr_weights = np.mean(reshaped_weights, axis=1)
            
            # Use uniform weights for LLMs
            optimal_llm_weights = np.ones(NUM_LLMS) / NUM_LLMS
        
        # Normalize weights
        if optimal_llm_weights is not None and np.sum(optimal_llm_weights) > 0:
            optimal_llm_weights = optimal_llm_weights / np.sum(optimal_llm_weights)

        if optimal_mr_weights is not None and np.sum(optimal_mr_weights) > 0:
            optimal_mr_weights = optimal_mr_weights / np.sum(optimal_mr_weights)
        
        self.optimal_mr_weights = optimal_mr_weights
        self.optimal_llm_weights = optimal_llm_weights

        
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

        print(llm_preds)
        confidences = self.get_pcs_label(llm_preds)
        label_confidences = {}
        for i in range(len(self.possible_labels)):
            label_confidences[self.possible_labels[i]] = confidences[i]
        return label_confidences

