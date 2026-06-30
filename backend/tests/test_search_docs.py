from unittest.mock import patch, MagicMock
from tools.search_docs import search_docs


def make_mock_match(text, source, page, score=0.9):
    match = MagicMock()
    match.metadata = {"text": text, "source": source, "page": page}
    match.score = score
    return match


def test_search_docs_returns_formatted_results():
    mock_embedding = [0.1] * 768
    mock_matches = [
        make_mock_match("Returns within 30 days are accepted.", "returns_policy.pdf", 1),
        make_mock_match("Products must be unused.", "returns_policy.pdf", 2),
    ]
    mock_query_result = MagicMock()
    mock_query_result.matches = mock_matches

    with patch("tools.search_docs.embed_text", return_value=mock_embedding), \
         patch("tools.search_docs._get_index") as mock_index_fn:

        mock_index = MagicMock()
        mock_index.query.return_value = mock_query_result
        mock_index_fn.return_value = mock_index

        results = search_docs("return policy")

    assert len(results) == 2
    assert results[0]["text"] == "Returns within 30 days are accepted."
    assert results[0]["source"] == "returns_policy.pdf"
    assert results[0]["page"] == 1


def test_search_docs_empty_results():
    mock_query_result = MagicMock()
    mock_query_result.matches = []

    with patch("tools.search_docs.embed_text", return_value=[0.1] * 768), \
         patch("tools.search_docs._get_index") as mock_index_fn:

        mock_index = MagicMock()
        mock_index.query.return_value = mock_query_result
        mock_index_fn.return_value = mock_index

        results = search_docs("something obscure")

    assert results == []
