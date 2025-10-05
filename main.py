from Core.Constants import Constants
#from Models.OpenAICompletion import OpenAICompletion
#from Models.GeminiCompletion import GeminiCompletion
#from VettingService import VettingService
from MLModel.TransitVetter import TransitVetter
def main():
    #gemini = GeminiCompletion()
   

   # vettingService = VettingService()
    
    classicalClassifier = TransitVetter()
    with open("Combined_Exoplanet_Data.csv", "rb") as f:
      metrics =  classicalClassifier.train_and_eval(f)
      print("\n=== classification report ===\n", metrics["report"])
      print("labels:", metrics["labels"])
      print("confusion_matrix:", metrics["confusion_matrix"])


    print('Welcome to the Vetting Tool with LLM!')

if __name__ == "__main__":
    main()