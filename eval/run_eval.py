import sys
sys.path.insert(0, ".")
from eval.evaluate import load_questions, run_evaluation, summarize
from retrieval.vanilla_rag import run as vanilla_run
from retrieval.graph_retriever import run as graph_run
from retrieval.hippo_retriever import run as hippo_run
from agent.sage import run as sage_run

questions = load_questions()
print(f"Total questions: {len(questions)}")

configs = {
    "vanilla_rag": vanilla_run,
    "graph_rag":   graph_run,
    "hippo_rag":   hippo_run,
    "sage":        sage_run,
}

results = run_evaluation(
    configs=configs,
    questions=questions,
    output_file="eval/results.json",
    max_questions=None
)

summarize("eval/results.json")