from framework.pipeline import run_pipeline
import llms.MODELS as MODELS

if __name__ == "__main__":
    run_pipeline(dataset_path="datasets/sample_dataset.json", model=MODELS.OPENAI_GPT41)