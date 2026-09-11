"""
Web Chatbot Giao Diện Thân Thiện & Đẹp Mắt
K4 Day 01 — Trợ Lý AI Tương Tác
Hỗ trợ: Chế độ Mock AI thông minh (Không cần API key / Không tốn credit) & Chế độ OpenAI API thật
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

# Nạp cấu hình và các hàm từ template.py
from template import (
    OPENAI_MINI_MODEL,
    OPENAI_MODEL,
    PRICING_PER_1K_TOKENS,
    compare_models,
    count_tokens,
    estimate_cost,
    retry_with_backoff,
)

PORT = 8000


def get_mock_reply(prompt: str, persona: str, temperature: float, model: str, history: list) -> str:
    p_lower = prompt.lower()
    persona_lower = persona.lower()

    is_teacher = "tiểu học" in persona_lower or "trẻ" in persona_lower
    is_expert = "chuyên gia" in persona_lower or "tài chính" in persona_lower
    is_coder = "coder" in persona_lower or "python" in persona_lower or "developer" in persona_lower

    if any(k in p_lower for k in ["so sánh", "compare", "gpt-4o vs", "gpt-4o và"]):
        if is_teacher:
            return (
                "Chào con nhé! Thầy giải thích giống như hai chiếc xe nhé:\n\n"
                "🚗 **GPT-4o** giống như một chiếc xe buýt lớn chở được rất nhiều đồ và rất khỏe, "
                "giúp giải những câu đố hóc búa nhất. Nhưng xe này cần nhiều xăng hơn một chút!\n\n"
                "🛴 **GPT-4o-mini** giống như một chiếc xe scooter nhỏ gọn, chạy siêu nhanh, "
                "rất nhẹ nhàng và tiết kiệm xăng gấp 16 lần! Hàng ngày đi dạo thì dùng scooter là tuyệt nhất con nhé!"
            )
        elif is_expert:
            return (
                "Phân tích so sánh định lượng giữa **GPT-4o** và **GPT-4o-mini** trên môi trường Production:\n\n"
                "1. **Chi phí (Unit Economics)**:\n"
                "   • GPT-4o: $0.0025/1K input, $0.010/1K output.\n"
                "   • GPT-4o-mini: $0.00015/1K input, $0.0006/1K output.\n"
                "   &rarr; Chi phí biên của Mini giảm xấp xỉ **16.7 lần**, tối ưu ROI đáng kể cho các dịch vụ High-QPS.\n\n"
                "2. **Độ trễ (Latency & TTFT)**:\n"
                "   • Mini tối ưu hóa thời gian giải mã (decoding time), TTFT nhanh hơn ~40-60%.\n\n"
                "3. **Kiến nghị kiến trúc (Model Cascading)**:\n"
                "   • 90% truy vấn cơ bản (phân loại intent, tóm tắt) định tuyến tới Mini.\n"
                "   • 10% tác vụ phức tạp (pháp lý, audit logic, code refactoring) định tuyến về GPT-4o."
            )
        elif is_coder:
            return (
                "So sánh kỹ thuật giữa 2 models:\n\n"
                "```python\n"
                "# Pricing per 1K output tokens (USD)\n"
                "COST = {'gpt-4o': 0.010, 'gpt-4o-mini': 0.0006}\n"
                "cost_ratio = COST['gpt-4o'] / COST['gpt-4o-mini']  # 16.67x\n"
                "```\n\n"
                "- **GPT-4o**: Phù hợp Complex Code Generation, Multi-step Refactoring, AST analysis.\n"
                "- **GPT-4o-mini**: Tối ưu cho Linting, Docstring generation, CRUD APIs, Test case stubbing. Tốc độ cao, rẻ gấp ~17 lần."
            )
        else:
            return (
                "Dưới đây là so sánh toàn diện giữa **GPT-4o** và **GPT-4o-mini**:\n\n"
                "| Tiêu chí | GPT-4o (Full) | GPT-4o-mini |\n"
                "| :--- | :--- | :--- |\n"
                "| **Chất lượng** | Suy luận logic phức tạp, giải quyết bài toán khó | Xuất sắc cho các tác vụ thông thường |\n"
                "| **Tốc độ** | Nhanh, độ trễ trung bình | Cực nhanh, phản hồi tức thì |\n"
                "| **Giá Input** | $0.0025 / 1K token | $0.00015 / 1K token |\n"
                "| **Giá Output** | $0.010 / 1K token | $0.0006 / 1K token (rẻ hơn 16.7x) |\n\n"
                "💡 **Khuyên dùng**: Chọn **GPT-4o-mini** làm mặc định cho hầu hết ứng dụng chat để tiết kiệm chi phí; chỉ chuyển sang **GPT-4o** khi cần tư duy suy luận sâu."
            )

    elif any(k in p_lower for k in ["token", "đếm token", "tiktoken"]):
        if is_teacher:
            return (
                "Con hãy tưởng tượng câu từ giống như những toa tàu Lego nhé!\n\n"
                "🚂 Khi máy tính đọc chữ, nó không đọc cả câu dài mà bẻ nhỏ ra thành từng mảnh ghép gọi là **Token**. "
                "Ví dụ từ 'mèo' có thể là 1 mảnh, nhưng từ có dấu phức tạp sẽ phải ghép từ 2 mảnh Lego lại đấy!"
            )
        elif is_expert:
            return (
                "**Bản chất cơ chế Tokenization trong LLM**:\n\n"
                "• Tokenizer sử dụng thuật toán Byte-Pair Encoding (BPE) (ví dụ bộ mã hóa `o200k_base` trên GPT-4o). "
                "Token không phải là một từ đơn lẻ mà là một chuỗi subwords hoặc chuỗi bytes.\n"
                "• **Hiện tượng lạm phát token tiếng Việt**: Do từ điển BPE được huấn luyện áp đảo trên ngữ liệu tiếng Anh, "
                "các ký tự có dấu thanh Unicode tiếng Việt thường bị phân tách thành 2-3 bytes độc lập. "
                "Hệ quả là cùng một dung lượng văn bản, tiếng Việt tiêu tốn token cao hơn tiếng Anh từ 20%–35%."
            )
        else:
            return (
                "Trong LLM, **Token** là đơn vị xử lý văn bản cơ bản nhất:\n\n"
                "1. **Quy đổi cơ bản**: Trung bình với tiếng Anh, 1 token ≈ 0.75 từ (~4 ký tự).\n"
                "2. **Thư viện chính thức**: `tiktoken` của OpenAI dùng để đếm chính xác số lượng token mà model xử lý.\n"
                "3. **Chi phí hai chiều**: Giá token đầu vào (Input/Prompt) luôn rẻ hơn giá token đầu ra (Output/Completion) "
                "bởi vì quá trình sinh token đầu ra đòi hỏi tính toán autoregressive từng bước một (sequential generation)."
            )

    elif any(k in p_lower for k in ["backoff", "retry", "lỗi", "exponential"]):
        return (
            "**Exponential Backoff (Thử lại lũy thừa nhân đôi)**:\n\n"
            "• **Cách hoạt động**: Khi gặp lỗi mạng hoặc server báo `429 Too Many Requests`, client không retry dồn dập "
            "mà chờ thời gian tăng dần: `0.1s &rarr; 0.2s &rarr; 0.4s &rarr; 0.8s`.\n"
            "• **Tại sao cần thiết?**: Tránh hiện tượng **Thundering Herd Problem** (Bầy đàn giẫm đạp) — khi hàng nghìn "
            "thiết bị cùng retry vào đúng một giây cố định, server vừa kịp mở lại sẽ lập tức bị nghẽn và sập tiếp."
        )

    elif any(k in p_lower for k in ["việt nam", "vietnam", "thú vị", "sự thật"]):
        return (
            "Một sự thật thú vị về Việt Nam 🇻🇳:\n\n"
            "• **Hang Sơn Đoòng** (Quảng Bình) là hang động tự nhiên lớn nhất thế giới, có hệ sinh thái rừng nhiệt đới "
            "ngay bên trong lòng hang với mây mù tự tạo và tòa nhà 40 tầng có thể nằm lọt bên trong.\n"
            "• Việt Nam cũng là **quốc gia xuất khẩu cà phê lớn thứ hai thế giới** (đứng đầu về cà phê Robusta) "
            "và sở hữu hơn 3.260 km bờ biển trải dài hình chữ S tuyệt đẹp!"
        )

    elif any(k in p_lower for k in ["blockchain", "chuỗi khối"]):
        if is_teacher:
            return (
                "Blockchain giống như **cuốn sổ ghi điểm chung của cả lớp**!\n\n"
                "Mỗi bạn trong lớp đều cầm một bản sao y hệt nhau. Mỗi khi có bạn ghi điểm, tất cả các bạn cùng đồng thanh ghi vào sổ của mình. "
                "Vì ai cũng giữ một quyển sổ giống nhau nên không một ai có thể lén tẩy xóa hay gian lận được cả!"
            )
        else:
            return (
                "**Blockchain (Chuỗi khối)** là một cơ sở dữ liệu sổ cái phân tán (Distributed Ledger Technology):\n\n"
                "1. **Cấu trúc dữ liệu**: Các giao dịch được đóng gói thành các khối (Block) và liên kết với nhau bằng mã băm mật mã học (Cryptographic Hash Pointer).\n"
                "2. **Cơ chế đồng thuận**: Sử dụng Proof of Work (PoW) hoặc Proof of Stake (PoS) để các node mạng đạt được thỏa thuận mà không cần bên thứ ba tín nhiệm.\n"
                "3. **Tính chất cốt lõi**: Bất biến (Immutability), phi tập trung (Decentralization) và minh bạch (Transparency)."
            )

    elif any(k in p_lower for k in ["chào", "hello", "hi", "bạn là ai"]):
        return (
            f"Xin chào bạn! Tôi là trợ lý AI đang hoạt động ở **Chế độ Mock thông minh**.\n\n"
            f"• **Vai trò**: {persona}\n"
            f"• **Model giả lập**: `{model}` (Temperature: {temperature})\n\n"
            f"Tôi được trang bị sẵn toàn bộ kiến thức của **K4 Lab 01 (LLM Foundation)**, "
            f"sẵn sàng giải đáp mọi thắc mắc của bạn về API, Token, Streaming, Retry và Prompt Engineering mà **hoàn toàn miễn phí**, không lo lỗi credit hay quota!"
        )

    else:
        return (
            f"Trợ lý AI (Mock Mode - {model}):\n\n"
            f"Tôi đã tiếp nhận câu hỏi của bạn: **\"{prompt}\"**.\n\n"
            f"Theo định hướng Persona: _{persona}_\n\n"
            f"Hệ thống đã duy trì chính xác {len(history)//2} lượt lịch sử hội thoại trước đó và áp dụng mức ngẫu nhiên Temperature = {temperature}. "
            f"Bạn có thể thử nghiệm thêm các câu hỏi khác về nội dung bài lab nhé!"
        )


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="vi" class="h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Assistant &mdash; K4 Lab 01</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Plus Jakarta Sans', sans-serif; }
    .custom-scrollbar::-webkit-scrollbar { width: 6px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #334155; border-radius: 9999px; }
    .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #475569; }
    @keyframes pulse-dot {
      0%, 100% { opacity: 0.3; transform: scale(0.8); }
      50% { opacity: 1; transform: scale(1.1); }
    }
    .typing-dot { animation: pulse-dot 1.2s infinite ease-in-out; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
  </style>
</head>
<body class="h-full bg-slate-950 text-slate-100 flex flex-col antialiased selection:bg-indigo-500 selection:text-white">

  <!-- Top Navigation -->
  <header class="bg-slate-900/80 backdrop-blur-md border-b border-slate-800 px-6 py-3.5 flex items-center justify-between z-10">
    <div class="flex items-center gap-3.5">
      <div class="relative">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/25">
          <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
          </svg>
        </div>
        <span class="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-500 border-2 border-slate-900 rounded-full"></span>
      </div>
      <div>
        <div class="flex items-center gap-2">
          <h1 class="font-bold text-base text-slate-100">K4 AI Assistant</h1>
          <button id="mode-badge" onclick="toggleMode()" class="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 transition-all flex items-center gap-1">
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
            <span id="mode-text">NVIDIA NIM (Llama 3.2 AI Thật)</span>
          </button>
        </div>
        <p class="text-xs text-slate-400">Trợ giảng AI thông minh &middot; LLM Foundation</p>
      </div>
    </div>

    <!-- Live Stats Bar -->
    <div class="hidden md:flex items-center gap-6 px-4 py-1.5 rounded-xl bg-slate-800/60 border border-slate-700/60 text-xs">
      <div class="flex items-center gap-1.5 text-slate-300">
        <svg class="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
        <span>Lượt hỏi:</span>
        <span id="stat-turns" class="font-bold text-indigo-400">0</span>
      </div>
      <div class="h-3 w-px bg-slate-700"></div>
      <div class="flex items-center gap-1.5 text-slate-300">
        <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"/></svg>
        <span>Tokens:</span>
        <span id="stat-tokens" class="font-bold text-emerald-400">0</span>
      </div>
      <div class="h-3 w-px bg-slate-700"></div>
      <div class="flex items-center gap-1.5 text-slate-300">
        <svg class="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        <span>Chi phí:</span>
        <span id="stat-cost" class="font-bold text-amber-400">$0.0000</span>
      </div>
    </div>

    <!-- Actions -->
    <div class="flex items-center gap-2">
      <button onclick="clearChat()" class="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors" title="Xóa lịch sử chat">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
      </button>
      <button onclick="toggleSettings()" class="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors" title="Cài đặt Persona & Tham số">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
      </button>
    </div>
  </header>

  <!-- Main Chat Container -->
  <main class="flex-1 flex flex-col max-w-5xl w-full mx-auto p-4 sm:p-6 overflow-hidden">
    <!-- Messages Scroll Area -->
    <div id="chat-box" class="flex-1 overflow-y-auto space-y-5 pr-2 custom-scrollbar">
      
      <!-- Welcome Message -->
      <div class="flex items-start gap-3.5">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-violet-500 flex items-center justify-center shrink-0 shadow-md">
          <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[85%] sm:max-w-2xl text-sm leading-relaxed shadow-sm">
          <p class="font-semibold text-indigo-400 mb-1">Chào bạn! Tôi là trợ lý AI khóa học K4.</p>
          <p class="text-slate-300">Đã kết nối thành công với <strong class="text-emerald-400">NVIDIA NIM (Llama 3.2 AI Thật)</strong>! Bạn có thể đặt bất kỳ câu hỏi nào, mô hình sẽ suy luận và stream câu trả lời theo thời gian thực.</p>
        </div>
      </div>

    </div>

    <!-- Suggested Quick Prompts -->
    <div id="suggestions" class="pt-4 pb-2 flex items-center gap-2 overflow-x-auto custom-scrollbar text-xs">
      <span class="text-slate-500 shrink-0 font-medium">Gợi ý:</span>
      <button onclick="sendQuickPrompt(this)" class="shrink-0 px-3 py-1.5 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 transition-all shadow-sm">
        ⚡ So sánh GPT-4o vs GPT-4o-mini
      </button>
      <button onclick="sendQuickPrompt(this)" class="shrink-0 px-3 py-1.5 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 transition-all shadow-sm">
        💡 Giải thích Token trong LLM
      </button>
      <button onclick="sendQuickPrompt(this)" class="shrink-0 px-3 py-1.5 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 transition-all shadow-sm">
        🔄 Tại sao cần Exponential Backoff?
      </button>
      <button onclick="sendQuickPrompt(this)" class="shrink-0 px-3 py-1.5 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 transition-all shadow-sm">
        🇻🇳 Một sự thật thú vị về Việt Nam
      </button>
    </div>

    <!-- Input Bar -->
    <div class="pt-2">
      <form id="chat-form" onsubmit="handleSend(event)" class="relative flex items-end gap-2 bg-slate-900 border border-slate-800 focus-within:border-indigo-500/80 rounded-2xl p-2.5 shadow-xl transition-all">
        <textarea
          id="user-input"
          rows="1"
          placeholder="Nhập tin nhắn... (Nhấn Enter để gửi, Shift+Enter để xuống dòng)"
          class="flex-1 bg-transparent border-0 resize-none text-slate-200 placeholder-slate-500 text-sm focus:ring-0 focus:outline-none px-2 py-1 max-h-32 custom-scrollbar leading-relaxed"
          onkeydown="handleKeyDown(event)"
          oninput="autoResize(this)"
        ></textarea>
        <button
          id="send-btn"
          type="submit"
          class="p-2.5 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white shadow-md shadow-indigo-600/30 transition-all disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/></svg>
        </button>
      </form>
      <div class="flex justify-between items-center text-[11px] text-slate-500 mt-2 px-1">
        <span>Cửa sổ trượt: Tối đa 3 lượt hội thoại gần nhất &middot; Tokenizer tiktoken thật</span>
        <button onclick="toggleMode()" class="hover:text-indigo-400 underline">Đổi chế độ: <span id="mode-status-text">Mock Model</span></button>
      </div>
    </div>
  </main>

  <!-- Settings Modal -->
  <div id="settings-modal" class="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
    <div class="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md overflow-hidden shadow-2xl animate-in fade-in zoom-in duration-200">
      <div class="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <svg class="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"/></svg>
          <h2 class="font-bold text-sm text-slate-100">Cấu hình Persona & Tham số</h2>
        </div>
        <button onclick="toggleSettings()" class="text-slate-400 hover:text-slate-200">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>

      <div class="p-5 space-y-4 text-xs">
        <!-- Execution Mode -->
        <div class="p-3 rounded-xl bg-slate-800/80 border border-slate-700/80 flex items-center justify-between">
          <div>
            <span class="font-semibold text-slate-200 block">Chế độ Mock</span>
            <span class="text-[11px] text-slate-400">Không cần API key / Không tốn credit</span>
          </div>
          <input type="checkbox" id="mock-checkbox" checked onchange="handleMockToggle(this.checked)" class="w-4 h-4 accent-indigo-500 rounded cursor-pointer">
        </div>

        <!-- Persona Preset -->
        <div>
          <label class="block font-medium text-slate-300 mb-1.5">Mẫu Persona (System Prompt)</label>
          <select id="persona-select" onchange="changePersonaPreset(this.value)" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500">
            <option value="assistant">Trợ giảng AI thân thiện (Mặc định)</option>
            <option value="teacher">Giáo viên tiểu học (Đơn giản cho trẻ 8 tuổi)</option>
            <option value="expert">Chuyên gia kỹ thuật / Tài chính (Chuyên sâu)</option>
            <option value="coder">Senior Python Developer (Ngắn gọn, trọng tâm)</option>
          </select>
        </div>

        <!-- System Prompt Textarea -->
        <div>
          <label class="block font-medium text-slate-300 mb-1.5">Nội dung System Prompt</label>
          <textarea id="system-prompt" rows="3" class="w-full bg-slate-800 border border-slate-700 rounded-xl p-2.5 text-slate-200 text-xs focus:outline-none focus:border-indigo-500 resize-none">Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt, tập trung vào bản chất và kèm ví dụ minh họa súc tích.</textarea>
        </div>

        <!-- Temperature Slider -->
        <div>
          <div class="flex justify-between text-slate-300 mb-1">
            <span>Temperature: <strong id="temp-val" class="text-indigo-400">0.7</strong></span>
            <span class="text-slate-500">0.0 (Chính xác) &rarr; 1.5 (Sáng tạo)</span>
          </div>
          <input id="temp-slider" type="range" min="0" max="1.5" step="0.1" value="0.7" oninput="document.getElementById('temp-val').innerText = this.value" class="w-full accent-indigo-500">
        </div>

        <!-- Model Select -->
        <div>
          <label class="block font-medium text-slate-300 mb-1.5">Model sử dụng</label>
          <select id="model-select" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500">
            <option value="meta/llama-3.2-11b-vision-instruct" selected>Llama 3.2 (NVIDIA NIM - Đang hoạt động)</option>
            <option value="gpt-4o">GPT-4o (OpenAI)</option>
            <option value="gpt-4o-mini">GPT-4o-mini (OpenAI)</option>
          </select>
        </div>
      </div>

      <div class="px-5 py-3 bg-slate-800/40 border-t border-slate-800 flex justify-end">
        <button onclick="toggleSettings()" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold shadow-md transition-all">Lưu cấu hình</button>
      </div>
    </div>
  </div>

  <script>
    let chatHistory = [];
    let stats = { turns: 0, tokens: 0, cost: 0.0 };
    let isGenerating = false;
    let isMockMode = false; // Đã cấu hình key NVIDIA NIM thành công, chạy Model Thật!

    const personaPresets = {
      assistant: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt, tập trung vào bản chất và kèm ví dụ minh họa súc tích.",
      teacher: "Bạn là giáo viên tiểu học, giải thích mọi khái niệm thật đơn giản, gần gũi cho trẻ 8 tuổi, dùng các ví dụ sinh động trong đời sống.",
      expert: "Bạn là chuyên gia tài chính và công nghệ cao cấp, trả lời chuyên sâu, sử dụng thuật ngữ kỹ thuật chính xác và phân tích đa chiều.",
      coder: "Bạn là Senior Python Software Architect, trả lời cực kỳ ngắn gọn, trực diện vào giải pháp code, tuân thủ Clean Code và PEP 8."
    };

    function toggleMode() {
      isMockMode = !isMockMode;
      updateModeUI();
    }

    function handleMockToggle(checked) {
      isMockMode = checked;
      updateModeUI();
    }

    function updateModeUI() {
      const badge = document.getElementById('mode-badge');
      const text = document.getElementById('mode-text');
      const statusText = document.getElementById('mode-status-text');
      const checkbox = document.getElementById('mock-checkbox');
      checkbox.checked = isMockMode;

      if (isMockMode) {
        badge.className = "text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 transition-all flex items-center gap-1";
        text.innerText = "Mock Model (Miễn phí)";
        statusText.innerText = "Mock Model (Miễn phí)";
      } else {
        badge.className = "text-xs font-semibold px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 hover:bg-indigo-500/20 transition-all flex items-center gap-1";
        text.innerText = "OpenAI API Thật";
        statusText.innerText = "OpenAI API Thật";
      }
    }

    function changePersonaPreset(val) {
      document.getElementById('system-prompt').value = personaPresets[val] || personaPresets.assistant;
    }

    function toggleSettings() {
      const modal = document.getElementById('settings-modal');
      modal.classList.toggle('hidden');
    }

    function autoResize(textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = (textarea.scrollHeight) + 'px';
    }

    function handleKeyDown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        document.getElementById('chat-form').dispatchEvent(new Event('submit'));
      }
    }

    function sendQuickPrompt(btn) {
      const text = btn.innerText.replace(/^[^\w\s]+/, '').trim();
      document.getElementById('user-input').value = text;
      document.getElementById('chat-form').dispatchEvent(new Event('submit'));
    }

    function clearChat() {
      chatHistory = [];
      stats = { turns: 0, tokens: 0, cost: 0.0 };
      updateStats();
      const chatBox = document.getElementById('chat-box');
      chatBox.innerHTML = `
        <div class="flex items-start gap-3.5">
          <div class="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-violet-500 flex items-center justify-center shrink-0 shadow-md">
            <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
          </div>
          <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[85%] sm:max-w-2xl text-sm leading-relaxed shadow-sm">
            <p class="font-semibold text-indigo-400 mb-1">Lịch sử chat đã được làm mới.</p>
            <p class="text-slate-300">Bạn muốn tìm hiểu thêm điều gì hôm nay?</p>
          </div>
        </div>
      `;
    }

    function updateStats() {
      document.getElementById('stat-turns').innerText = stats.turns;
      document.getElementById('stat-tokens').innerText = stats.tokens.toLocaleString();
      document.getElementById('stat-cost').innerText = '$' + stats.cost.toFixed(4);
    }

    async function handleSend(e) {
      e.preventDefault();
      if (isGenerating) return;

      const input = document.getElementById('user-input');
      const text = input.value.trim();
      if (!text) return;

      input.value = '';
      input.style.height = 'auto';

      appendUserMessage(text);

      const botBubble = appendBotMessageBubble();
      isGenerating = true;
      document.getElementById('send-btn').disabled = true;

      const payload = {
        message: text,
        history: chatHistory,
        persona: document.getElementById('system-prompt').value,
        temperature: parseFloat(document.getElementById('temp-slider').value),
        model: document.getElementById('model-select').value,
        use_mock: isMockMode
      };

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Yêu cầu thất bại: ' + response.statusText);

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let fullReply = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          fullReply += chunk;
          botBubble.innerHTML = formatMarkdown(fullReply);
          scrollToBottom();
        }

        // Cập nhật history (tối đa 3 lượt = 6 messages)
        chatHistory.push({ role: 'user', content: text });
        chatHistory.push({ role: 'assistant', content: fullReply });
        if (chatHistory.length > 6) chatHistory = chatHistory.slice(-6);

        // Lấy thống kê lượt gọi
        const statsRes = await fetch('/api/stats', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: text, response: fullReply, model: payload.model })
        });
        if (statsRes.ok) {
          const st = await statsRes.json();
          stats.turns += 1;
          stats.tokens += st.tokens;
          stats.cost += st.cost;
          updateStats();
        }

      } catch (err) {
        botBubble.innerHTML = `<span class="text-rose-400">⚠️ Lỗi: ${err.message}</span>`;
      } finally {
        isGenerating = false;
        document.getElementById('send-btn').disabled = false;
        scrollToBottom();
      }
    }

    function appendUserMessage(text) {
      const chatBox = document.getElementById('chat-box');
      const div = document.createElement('div');
      div.className = 'flex items-start justify-end gap-3';
      div.innerHTML = `
        <div class="bg-gradient-to-r from-indigo-600 to-indigo-500 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 max-w-[85%] sm:max-w-2xl text-sm leading-relaxed shadow-md break-words">
          ${escapeHtml(text)}
        </div>
      `;
      chatBox.appendChild(div);
      scrollToBottom();
    }

    function appendBotMessageBubble() {
      const chatBox = document.getElementById('chat-box');
      const div = document.createElement('div');
      div.className = 'flex items-start gap-3.5';
      div.innerHTML = `
        <div class="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-violet-500 flex items-center justify-center shrink-0 shadow-md">
          <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
        </div>
        <div class="bot-content bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[85%] sm:max-w-2xl text-sm leading-relaxed shadow-sm text-slate-200 break-words">
          <span class="inline-flex gap-1 py-1">
            <span class="w-2 h-2 rounded-full bg-indigo-400 typing-dot"></span>
            <span class="w-2 h-2 rounded-full bg-indigo-400 typing-dot"></span>
            <span class="w-2 h-2 rounded-full bg-indigo-400 typing-dot"></span>
          </span>
        </div>
      `;
      chatBox.appendChild(div);
      scrollToBottom();
      return div.querySelector('.bot-content');
    }

    function scrollToBottom() {
      const chatBox = document.getElementById('chat-box');
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    function escapeHtml(str) {
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    function formatMarkdown(text) {
      let html = escapeHtml(text);
      // Code blocks
      html = html.replace(/```([a-z]*)\n([\s\S]*?)```/g, '<pre class="bg-slate-950 p-3 rounded-lg border border-slate-800 overflow-x-auto text-xs font-mono text-emerald-400 my-2"><code>$2</code></pre>');
      // Inline code
      html = html.replace(/`([^`]+)`/g, '<code class="bg-slate-800 text-indigo-300 px-1.5 py-0.5 rounded text-xs font-mono">$1</code>');
      // Bold
      html = html.replace(/\*\*([^\*]+)\*\*/g, '<strong class="text-white font-semibold">$1</strong>');
      // Italic
      html = html.replace(/_([^_]+)_/g, '<em class="text-slate-400">$1</em>');
      // Lists
      html = html.replace(/•\s*([^\n]+)/g, '<div class="flex items-start gap-2 my-1"><span class="text-indigo-400 font-bold">•</span><span>$1</span></div>');
      // Line breaks
      html = html.replace(/\n/g, '<br>');
      return html;
    }
  </script>
</body>
</html>
"""


class ChatRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        url = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(body) if body else {}

        if url.path == "/api/chat":
            self.handle_chat_stream(data)
        elif url.path == "/api/stats":
            self.handle_stats(data)
        else:
            self.send_error(404, "Endpoint not found")

    def handle_chat_stream(self, data):
        prompt = data.get("message", "")
        history = data.get("history", [])
        persona = data.get("persona", "Bạn là trợ giảng thân thiện của khóa AI.")
        temperature = float(data.get("temperature", 0.7))
        model = data.get("model", OPENAI_MODEL)
        use_mock = data.get("use_mock", True)

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        # 1. Chế độ Mock Model (Khuyên dùng khi không có API credit)
        if use_mock:
            mock_reply = get_mock_reply(prompt, persona, temperature, model, history)
            # Stream mượt mà từng từ
            words = mock_reply.split(" ")
            for i, word in enumerate(words):
                chunk = (word if i == 0 else " " + word)
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.025)
            return

        # 2. Chế độ API thật (nếu có key & còn credit)
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or "your-key-here" in api_key:
            fallback = get_mock_reply(prompt, persona, temperature, model, history)
            note = "\n\n*(Lưu ý: Đã tự động chuyển sang Mock Model do chưa có API key trong .env)*"
            for word in (fallback + note).split(" "):
                chunk = word + " "
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.025)
            return

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL", None),
            )
            messages = [{"role": "system", "content": persona}] + history + [{"role": "user", "content": prompt}]

            stream = retry_with_backoff(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    self.wfile.write(delta.encode("utf-8"))
                    self.wfile.flush()

        except Exception as e:
            # Nếu API thật lỗi (ví dụ 429 quota exhausted), tự động chuyển sang Mock Model
            err_str = str(e)
            if "insufficient_quota" in err_str or "429" in err_str:
                reason = "API của bạn đã hết credit (lỗi 429). Hệ thống đã tự động chuyển sang Mock Model thông minh để tiếp tục trải nghiệm không gián đoạn."
            else:
                reason = f"Kết nối API gặp lỗi ({e}). Đã tự động chuyển sang Mock Model."

            fallback = get_mock_reply(prompt, persona, temperature, model, history)
            prefix = f"⚠️ *[{reason}]*\n\n"
            for word in (prefix + fallback).split(" "):
                chunk = word + " "
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.02)

    def handle_stats(self, data):
        prompt = data.get("prompt", "")
        response = data.get("response", "")
        model = data.get("model", OPENAI_MODEL)

        in_tok = count_tokens(prompt, model=model)
        out_tok = count_tokens(response, model=model)
        cost_info = estimate_cost(prompt, response, model=model)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        res_data = {
            "tokens": in_tok + out_tok,
            "cost": cost_info["total_cost"],
        }
        self.wfile.write(json.dumps(res_data).encode("utf-8"))

    def log_message(self, format, *args):
        return


def run_server(port=PORT):
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), ChatRequestHandler) as httpd:
        print("=" * 60)
        print(f"🚀 GIAO DIỆN CHATBOT (CHẾ ĐỘ MOCK MẶC ĐỊNH) ĐÃ BẬT:")
        print(f"👉 http://localhost:{port}")
        print("=" * 60)
        print("• Không tốn credit API & không lo lỗi 429.")
        print("• Nhấn Ctrl+C trong terminal để dừng.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nĐã tắt máy chủ.")


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(p)
