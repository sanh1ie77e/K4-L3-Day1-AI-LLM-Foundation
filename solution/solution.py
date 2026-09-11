"""
K4 — Ngày 1: Khám Phá LLM API (4 tiếng)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Tên model có thể đổi qua .env — ví dụ khi dùng NVIDIA NIM miễn phí
# (xem LAB_GUIDE.md, Phụ lục B). Không đặt gì trong .env thì mặc định OpenAI.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: phút 60–100)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI  # import TRONG hàm — bắt buộc, để test có thể mock

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    end = time.perf_counter()

    response_text = response.choices[0].message.content
    latency_seconds = end - start

    return response_text, latency_seconds

    # TODO: import OpenAI, tạo client, gọi chat.completions.create,
    #       đo start/end time, trả về (response_text, latency)
    raise NotImplementedError("Implement call_openai")


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    return call_openai(
        prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    # TODO: gọi call_openai với model=OPENAI_MINI_MODEL
    raise NotImplementedError("Implement call_openai_mini")


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    gpt4o_response, gpt4o_latency = call_openai(prompt)
    mini_response, mini_latency = call_openai_mini(prompt)

    # Ước lượng thô: 0.75 từ ≈ 1 token (Part 2 sẽ thay bằng tiktoken)
    prompt_words = len(prompt.split())
    response_words = len(gpt4o_response.split())

    estimated_input_tokens = prompt_words / 0.75
    estimated_output_tokens = response_words / 0.75

    pricing = PRICING_PER_1K_TOKENS.get(OPENAI_MODEL, {"input": 0.0, "output": 0.0})
    input_cost = (estimated_input_tokens / 1000) * pricing["input"]
    output_cost = (estimated_output_tokens / 1000) * pricing["output"]
    gpt4o_cost_estimate = input_cost + output_cost

    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate,
    }
    # TODO: gọi call_openai và call_openai_mini, ghép dict kết quả
    raise NotImplementedError("Implement compare_models")


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: phút 100–140)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI  # import TRONG hàm, giống call_openai

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    end = time.perf_counter()

    response_text = response.choices[0].message.content
    latency_seconds = end - start

    return response_text, latency_seconds
    # TODO: giống call_openai nhưng messages có thêm phần tử role="system"
    raise NotImplementedError("Implement chat_with_system_prompt")


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Phương án dự phòng khi encoding lỗi hoặc model lạ
        return max(1, len(text) // 4)
    # TODO: dùng tiktoken để đếm token, có fallback khi lỗi
    raise NotImplementedError("Implement count_tokens")


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(
    input_text: str,
    output_text: str,
    model: str = OPENAI_MODEL,
) -> dict:
    input_tokens = count_tokens(input_text, model)
    output_tokens = count_tokens(output_text, model)

    # .get(...) với giá trị dự phòng — KHÔNG dùng PRICING_PER_1K_TOKENS[model]
    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS.get(OPENAI_MODEL))

    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }
    # TODO: đếm token prompt/response, tra bảng giá, trả về dict 5 key
    raise NotImplementedError("Implement estimate_cost")


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: phút 150–190)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    history: list[dict] = []

    while True:
        user_input = input("Bạn: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        history.append({"role": "user", "content": user_input})

        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            stream=True,
        )

        print("Bot: ", end="", flush=True)
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            reply += delta
            print(delta, end="", flush=True)
        print()

        history.append({"role": "assistant", "content": reply})
        # Giữ 3 lượt gần nhất = 6 message (user+assistant mỗi lượt)
        history = history[-6:]

# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    func,
    *args,
    max_retries: int = 3,
    base_delay: float = 0.1,
    **kwargs,
):
    delay = base_delay
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2
    # TODO: vòng lặp retry với exponential backoff
    raise NotImplementedError("Implement retry_with_backoff")


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: phút 190–230)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input=input,
    max_turns: int = 10,
    model: str = OPENAI_MODEL,
) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    history: list[dict] = []
    total_tokens = 0
    total_cost = 0.0
    num_turns = 0

    while num_turns < max_turns:
        user_input = get_input("Bạn: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": persona}] + history

        def do_call():
            return client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

        stream = retry_with_backoff(do_call)

        print("Bot: ", end="", flush=True)
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            reply += delta
            print(delta, end="", flush=True)
        print()

        history.append({"role": "assistant", "content": reply})
        history = history[-6:]

        cost_info = estimate_cost(user_input, reply, model)
        total_tokens += cost_info["input_tokens"] + cost_info["output_tokens"]
        total_cost += cost_info["total_cost"]

        num_turns += 1

    return {
        "num_turns": num_turns,
        "history": history,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
    }
    # TODO (bonus): dựng chuỗi bảng và trả về
    raise NotImplementedError("Implement format_comparison_table")


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
                "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
