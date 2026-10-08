"""Streamlit UI for inspecting and comparing Hugging Face tokenizers."""

import streamlit as st

from tokenizer_utils import compare_tokenizers, load_tokenizer, parse_token_ids, sorted_vocab, vocab_stats


st.set_page_config(page_title="Tokenizer Viewer", layout="wide")


def show_vocab(vocab: dict[str, int], label: str) -> None:
    query = st.text_input(f"搜索 {label} token", key=f"search_{label}")
    if query:
        if query in vocab:
            st.success(f"精确匹配：{query} → {vocab[query]}")
        else:
            st.info("没有精确匹配。")
    limit = st.number_input(f"{label} 显示条数", min_value=1, max_value=1000, value=100, key=f"limit_{label}")
    rows = sorted_vocab(vocab)
    if query:
        rows = [row for row in rows if query in row["token"]]
    st.caption(f"匹配 {len(rows)} 条，显示前 {min(len(rows), limit)} 条")
    st.dataframe(rows[:limit], width="stretch", hide_index=True)


def viewer_page() -> None:
    st.write("查看词表、搜索 token，并测试编码与解码。路径可为本地 tokenizer 目录或 Hugging Face 模型 ID。")
    source = st.text_input("Tokenizer 路径或模型 ID", key="viewer_source")
    if st.button("加载 Tokenizer", key="load_viewer"):
        try:
            tokenizer = load_tokenizer(source)
        except Exception as exc:
            st.session_state.pop("viewer_loaded", None)
            st.error(f"加载失败：{exc}")
        else:
            st.session_state["viewer_loaded"] = (source.strip(), tokenizer)

    loaded = st.session_state.get("viewer_loaded")
    if not loaded:
        return
    loaded_source, tokenizer = loaded
    stats = vocab_stats(tokenizer)
    st.subheader(f"当前 Tokenizer：{loaded_source}")
    st.write(f"基础词表：{stats['base_vocab']}；额外 token：{stats['added_vocab']}；总计：{stats['total_vocab']}")
    first, second = st.columns(2)
    with first:
        show_vocab(tokenizer.get_vocab(), "完整词表")
    with second:
        show_vocab(tokenizer.get_added_vocab(), "额外词表")

    st.subheader("编码与解码")
    sample = st.text_input("待编码文本", value="这是一个待编码的句子", key="encode_text")
    ids = tokenizer.encode(sample)
    st.json({"tokens": tokenizer.convert_ids_to_tokens(ids), "ids": ids})
    raw_ids = st.text_input("待解码的 JSON 整数数组", value="[0, 1, 2]", key="decode_ids")
    try:
        decoded_ids = parse_token_ids(raw_ids)
        st.json({"tokens": tokenizer.convert_ids_to_tokens(decoded_ids), "text": tokenizer.decode(decoded_ids)})
    except (ValueError, OverflowError) as exc:
        st.error(str(exc))


def comparison_page() -> None:
    st.write("比较两个 tokenizer 的独有 token，以及同一 token 是否使用不同 ID。")
    left, right = st.columns(2)
    with left:
        first_source = st.text_input("Tokenizer 1 路径或模型 ID", key="compare_first")
    with right:
        second_source = st.text_input("Tokenizer 2 路径或模型 ID", key="compare_second")
    if st.button("比较 Tokenizer", key="load_comparison"):
        try:
            result = compare_tokenizers(load_tokenizer(first_source), load_tokenizer(second_source))
        except Exception as exc:
            st.session_state.pop("comparison_loaded", None)
            st.error(f"比较失败：{exc}")
        else:
            st.session_state["comparison_loaded"] = (first_source.strip(), second_source.strip(), result)

    loaded = st.session_state.get("comparison_loaded")
    if not loaded:
        return
    first_source, second_source, result = loaded
    st.subheader(f"{first_source} ↔ {second_source}")
    st.dataframe([
        {"tokenizer": first_source, **result["first_stats"]},
        {"tokenizer": second_source, **result["second_stats"]},
    ], width="stretch", hide_index=True)
    limit = st.number_input("每类差异显示条数", min_value=1, max_value=1000, value=100)
    for key, heading in (
        ("only_first", "仅 Tokenizer 1 包含"),
        ("only_second", "仅 Tokenizer 2 包含"),
        ("different_ids", "共同 token 的 ID 不同"),
    ):
        rows = result[key]
        st.subheader(f"{heading}：{len(rows)}")
        st.dataframe(rows[:limit], width="stretch", hide_index=True)


def main() -> None:
    st.title("🤗 Tokenizer Viewer")
    viewer, comparison = st.tabs(["词表查看与编解码", "词表比较"])
    with viewer:
        viewer_page()
    with comparison:
        comparison_page()


if __name__ == "__main__":
    main()
