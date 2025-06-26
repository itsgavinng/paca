from framework.stage_1 import run_stage_1   
from framework.stage_2 import run_stage_2
from framework.pipeline import run_pipeline
import llms.MODELS as MODELS

if __name__ == "__main__":
    # test_question = "Did Adam and Eve have a bellybutton?"
    # # result = run_stage_1(test_question, dagent_model=MODELS.FIREWORKS_LLAMA3_70B, cagent_model=MODELS.FIREWORKS_LLAMA3_70B)
    # result = run_stage_1(test_question, dagent_model=MODELS.OPENAI_GPT41, cagent_model=MODELS.OPENAI_GPT41)

    # print("\n" + "="*80)
    # print("FINAL RESULTS")
    # print("="*80)
    # print(f"Question: {test_question}")
    # print(f"Best Dimension: {result.best_dimension.name}")
    # print(f"Perspectives: {[p.value for p in result.perspectives]}")
    # print(f"Final Weights: {list(result.final_weights.values())}")
    
    
    # run_stage_2(test_question, result, model=MODELS.OPENAI_GPT41, k=2, n=4, theta_max=0.5, rho_null=0.3)
    
    run_pipeline(dataset_path="datasets/sample_dataset.json", model=MODELS.OPENAI_GPT41, t=2, k=2, n=4)