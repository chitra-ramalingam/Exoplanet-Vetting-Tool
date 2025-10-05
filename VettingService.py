import pandas as pd
import numpy as np
from DataLoader import VettingDataLoader
from DataUtils import DatasetSplitter, pseudo_label_from_row
from Evaluator import Evaluator
from Classifier.VetClasssifer import  VettingConfig, AstraVetClassifier
import MemoryDataLoader as mdl

class VettingService:
    def __init__(self):
        pass

    def runLLM(self, uploaded):
        # Load Brett's CSV
     
        df = mdl.load_uploaded_csv(uploaded)  # adjust chunk size to your RAM; or None for single-pass        # Load Bretts csv into Dataframes

        loader = VettingDataLoader()
        df2 = loader.add_lc_summary(df)

        # Train/test split the dataframe inot train and test sets
        splitter = DatasetSplitter(test_size=0.4, random_state=7)
        train_df, test_df = splitter.split(df2)

        # Configure Ollama Mistral (schema is built-in)
        cfg = VettingConfig(base_url="http://localhost:11434", model="mistral", temperature=0.2, timeout=30, few_shot_k=4)
        clf = AstraVetClassifier(cfg)
        clf.fit(train_df, few_shot_k=1)
        preds = clf.predict(test_df)

    # Announce total
      #  yield -1, {"total": len(test_df)}

    # Stream each row; forward deltas and final result to the UI
        # for idx, ev in clf.predict_stream(test_df):
        #     yield idx, ev
        
        preds = list(clf.predict(test_df))
        test_df["pred"] = [p["label"] for p in preds]
        test_df["pred_conf"] = [p.get("confidence") for p in preds]
        return test_df