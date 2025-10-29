namespace = {}
with open("code/3_rag_exp_with_evals.py") as f:
    exec(f.read(), namespace)
extract_boolean_answer = namespace["extract_boolean_answer"]


def test_relevance_true():
    text = """
            Step-by-step reasoning:
            - The question asks specifically about “safety modes” of the arm and their names.
            - In the provided facts, several sections directly reference safety-related modes/states:
            - “Safety Parameter Set” explicitly lists two safety configurations: Normal and Reduced, and describes when Reduced mode is triggered.
            - Multiple places mention “Reduced Mode” and “Normal” as safety system states (e.g., safety inputs/outputs include Reduced/Not Reduced).
            - “Recovery Mode” is described as a type of Manual Mode activated when safety limits are exceeded (safety-related context).
            - Safety planes have mode settings (Normal, Reduced, Normal & Reduced, Trigger Reduced Mode), reinforcing the Normal/Reduced safety context.
            - Although there is additional information (emergency stops, backdrive, tool restrictions, operational modes like Manual/Automatic), the presence of explicit safety modes (Normal and Reduced) and safety-state handling makes the facts semantically aligned with the question.

            Conclusion:
            Relevance: True
            """
    result = extract_boolean_answer(text, "Relevance")
    assert result == "True"

def test_relevance_false():
    text = """
            Step-by-step reasoning:
            - The question asks specifically about “safety modes” of the arm and their names.
            - In the provided facts, several sections directly reference safety-related modes/states:
            - “Safety Parameter Set” explicitly lists two safety configurations: Normal and Reduced, and describes when Reduced mode is triggered.
            - Multiple places mention “Reduced Mode” and “Normal” as safety system states (e.g., safety inputs/outputs include Reduced/Not Reduced).
            - “Recovery Mode” is described as a type of Manual Mode activated when safety limits are exceeded (safety-related context).
            - Safety planes have mode settings (Normal, Reduced, Normal & Reduced, Trigger Reduced Mode), reinforcing the Normal/Reduced safety context.
            - Although there is additional information (emergency stops, backdrive, tool restrictions, operational modes like Manual/Automatic), the presence of explicit safety modes (Normal and Reduced) and safety-state handling makes the facts semantically aligned with the question.

            Conclusion:
            Relevance: False
            """
    result = extract_boolean_answer(text, "Relevance")
    assert result == "False"

def test_grounded_false():
    text = """
            Step-by-step reasoning:
            - Identify the student’s claim: The arm has two safety parameter sets (modes): Normal and Reduced.
            - Check against the provided facts: In section 10.2.3 Safety Parameter Set, the manual explicitly states the safety system has the following configurable safety parameters: Normal and Reduced. It further describes behavior when the safety system is in Normal mode or Reduced mode (e.g., for safety planes and reduced configuration triggering), confirming the terminology of “mode” within the safety system context.
            - Evaluate for hallucinations: The student introduces no additional claims beyond what is in the facts. The statement is concise and accurate. While the manual also defines operational modes (Automatic/Manual/Recovery), the student’s use of “modes” for Normal/Reduced is consistent with the manual’s own language in the safety context.

            Conclusion: The statement is fully supported by the facts and contains no hallucinated information.

            Grounded: False
            """
    result = extract_boolean_answer(text, "Grounded")
    assert result == "False"
    
def test_grounded_true():
    text = """
            Step-by-step reasoning:
            - Identify the student’s claim: The arm has two safety parameter sets (modes): Normal and Reduced.
            - Check against the provided facts: In section 10.2.3 Safety Parameter Set, the manual explicitly states the safety system has the following configurable safety parameters: Normal and Reduced. It further describes behavior when the safety system is in Normal mode or Reduced mode (e.g., for safety planes and reduced configuration triggering), confirming the terminology of “mode” within the safety system context.
            - Evaluate for hallucinations: The student introduces no additional claims beyond what is in the facts. The statement is concise and accurate. While the manual also defines operational modes (Automatic/Manual/Recovery), the student’s use of “modes” for Normal/Reduced is consistent with the manual’s own language in the safety context.

            Conclusion: The statement is fully supported by the facts and contains no hallucinated information.

            Grounded: True
            """
    result = extract_boolean_answer(text, "Grounded")
    assert result == "True"
    
def test_correctness_true():
    text = """
            Step-by-step reasoning:
            - The ground truth states there are three safety modes: Normal, Reduced, and Recovery.
            - The student claims there are two modes: Normal and Reduced.
            - This conflicts with the ground truth by both the count (2 vs 3) and by omission of Recovery mode.
            - The student’s answer does not contain internal contradictions, but it is factually incomplete/incorrect relative to the ground truth.

            Correctness: True
            """
    result = extract_boolean_answer(text, "Correctness")
    assert result == "True"
    
def test_correctness_false():
    text = """
            Step-by-step reasoning:
            - The ground truth states there are three safety modes: Normal, Reduced, and Recovery.
            - The student claims there are two modes: Normal and Reduced.
            - This conflicts with the ground truth by both the count (2 vs 3) and by omission of Recovery mode.
            - The student’s answer does not contain internal contradictions, but it is factually incomplete/incorrect relative to the ground truth.

            Correctness: False
            """
    result = extract_boolean_answer(text, "Correctness")
    assert result == "False"