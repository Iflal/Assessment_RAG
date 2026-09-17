from __future__ import annotations

import unittest

from rag_generator.evaluation import EvalCase, _grade
from rag_generator.generation.answerer import AnswerResult
from rag_generator.retrieval.retriever import RetrievedChunk


class EvaluationTests(unittest.TestCase):
    def test_grader_normalizes_unicode_whitespace(self) -> None:
        case = EvalCase(
            name="size",
            question="Maximum size?",
            required_term_groups=(("25 mb",),),
        )
        source = RetrievedChunk(
            id="chunk-one",
            text="Maximum 25 MB",
            source="handbook.md",
            chunk_index=0,
            similarity=0.9,
        )
        result = AnswerResult("The maximum is 25\u202fMB [S1].", (source,))

        passed, _ = _grade(case, result)

        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
