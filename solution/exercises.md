# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> - Ở temperature 0.0, phản hồi mang tính tiền định (deterministic), lập luận nhất quán và tập trung vào các sự thật phổ biến nhất (như xuất khẩu cà phê hoặc hang Sơn Đoòng).
> - Khi temperature tăng lên 0.5 – 1.0, nội dung bắt đầu đa dạng hóa góc nhìn, câu văn linh hoạt và tự nhiên hơn.
> - Ở mức 1.5, câu chữ trở nên kém ổn định, xuất hiện các cách diễn đạt bất thường hoặc câu từ rời rạc do phân phối xác suất token bị làm phẳng quá mức (high entropy).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature thấp, khoảng từ 0.0 đến 0.3 (khuyến nghị 0.2). Chatbot chăm sóc khách hàng yêu cầu tính chính xác tuyệt đối, nhất quán về chính sách/giá cả và hạn chế tối đa hiện tượng ảo giác (hallucination). Mức nhiệt độ thấp giúp model ưu tiên lựa chọn các token có xác suất cao nhất, đảm bảo thông tin đáng tin cậy và tuân thủ đúng tài liệu hỗ trợ.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> - Ước tính chi phí: Tổng output token mỗi ngày là 10.000 × 3 × 350 = 10.500.000 tokens (10.500 đơn vị 1K token). Chi phí output của GPT-4o là 10.500 × $0.010 = $105/ngày (~$3.150/tháng), trong khi GPT-4o-mini là 10.500 × $0.0006 = $6.3/ngày (~$189/tháng). GPT-4o đắt hơn GPT-4o-mini xấp xỉ 16.67 lần.
> - Trường hợp GPT-4o xứng đáng: Các bài toán cần suy luận logic phức tạp, giải quyết tình huống pháp lý/kỹ thuật chuyên sâu, sinh mã nguồn chính xác hoặc trích xuất dữ liệu rủi ro cao mà sai sót nhỏ gây tổn thất lớn.
> - Trường hợp nên dùng mini: Chatbot trả lời câu hỏi thường gặp (FAQ), phân loại ý định người dùng (intent classification), tóm tắt nhanh văn bản ngắn hoặc các luồng tương tác thông thường với lưu lượng truy vấn lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> - Persona giáo viên tiểu học: Phản hồi ngắn gọn, giọng điệu ấm áp, từ ngữ gần gũi và dùng ví dụ ẩn dụ trực quan (ví dụ cuốn sổ chung của cả lớp ghi lại lượt mượn đồ chơi mà bạn nào cũng giữ một bản sao, không ai xóa sửa được).
> - Persona chuyên gia tài chính: Phản hồi chi tiết, văn phong học thuật, sử dụng nhiều thuật ngữ chuyên môn như sổ cái phân tán (distributed ledger), mã hóa băm mật mã học (cryptographic hashing), cơ chế đồng thuận (PoW/PoS) và tính bất biến (immutability).
> - Ảnh hưởng của system prompt: Đóng vai trò là tiền điều kiện dẫn dắt (conditioning context), trực tiếp định hình persona, cách xưng hô, độ phức tạp của ngôn từ và hướng triển khai ví dụ cho toàn bộ lời phản hồi của model.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> - Với đoạn văn tiếng Việt 100 từ, công thức ước lượng thô theo từ cho ra 100 / 0.75 ≈ 133 token, trong khi `tiktoken` (o200k_base / cl100k_base) thực tế mã hóa ra khoảng 160 – 175 token, chênh lệch khoảng 20% – 30%.
> - Tiếng Việt tốn nhiều token hơn tiếng Anh vì các thuật toán phân tách token (như BPE) được huấn luyện chủ yếu trên ngữ liệu tiếng Anh, nơi từ nguyên vẹn hoặc gốc từ phổ biến có sẵn mã token riêng. Với tiếng Việt, các ký tự có dấu thanh Unicode và cấu trúc từ đơn lập ghép tiếng thường không có trong từ điển token tĩnh, buộc tokenizer phải phân rã từ thành nhiều subwords hoặc các byte UTF-8 riêng lẻ, làm tăng tổng số token.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng có tương tác người dùng theo thời gian thực (real-time chat, trợ lý ảo, giao diện CLI tương tác), nơi chỉ số Time-to-First-Token (TTFT) đóng vai trò sống còn để người dùng không phải đối mặt với trạng thái "đóng băng" chờ đợi suốt hàng chục giây khi model sinh câu trả lời dài. Ngược lại, non-streaming lại là lựa chọn tối ưu và sạch sẽ hơn cho các quy trình xử lý nền (backend background jobs, batch processing), các pipeline đa agent cần parse trọn vẹn cấu trúc dữ liệu (như JSON Object hay Pydantic model) trước khi chuyển tiếp sang bước sau, hoặc các tác vụ kiểm thử và chấm điểm tự động.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp khoảng thời gian chờ tăng theo cấp số nhân (0.1s -> 0.2s -> 0.4s...), tạo ra không gian và thời gian giãn cách cần thiết để hệ thống máy chủ giảm tải và kịp thời hồi phục. Nếu hàng nghìn client cùng sử dụng một khoảng delay cố định (ví dụ 1 giây), toàn bộ các client sẽ đồng loạt gửi lại request vào đúng thời điểm chu kỳ tiếp theo, gây ra hiện tượng "thundering herd problem" (bầy đàn giẫm đạp). Điều này tạo ra các đỉnh lưu lượng đột biến liên tục theo từng nhịp sóng, khiến server tiếp tục sập và không bao giờ thoát khỏi trạng thái nghẽn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> - Persona lựa chọn: Trợ giảng lập trình AI thân thiện và súc tích.
> - System prompt: `"Bạn là trợ giảng lập trình AI thân thiện, luôn giải thích ngắn gọn bằng tiếng Việt, tập trung vào bản chất kỹ thuật và kèm ví dụ minh họa súc tích."`
> - Giải thích từ ngữ:
>   1. `"ngắn gọn"`: Giúp model tránh lan man, kiểm soát chặt chẽ số lượng output token sinh ra, giảm độ trễ phản hồi (latency) và tiết kiệm chi phí trên mỗi lượt gọi.
>   2. `"bằng tiếng Việt"`: Định hình ngôn ngữ cố định, ngăn chặn model chuyển ngữ không mong muốn sang tiếng Anh khi gặp các thuật ngữ kỹ thuật.
>   3. `"kèm ví dụ minh họa súc tích"`: Giúp người học nắm bắt trực quan khái niệm trừu tượng mà vẫn giữ context length của hội thoại ở mức tối ưu.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> - Hạn chế lớn nhất: Cơ chế cắt sliding window cố định 3 lượt cuối (`history[-6:]`) khiến trợ lý bị "mất trí nhớ" hoàn toàn về các thông tin quan trọng ban đầu (như tên người dùng, sở thích, hoặc các quy ước đã thiết lập từ các lượt hỏi trước).
> - Đề xuất cải thiện: Triển khai kỹ thuật "Memory Summarization" (Tóm tắt hội thoại).
> - Cách triển khai: Khi `history` vượt quá ngưỡng quy định (ví dụ > 6 messages), trước khi cắt bỏ các message cũ nhất, ta gọi một model nhanh và rẻ (như GPT-4o-mini) với prompt yêu cầu tóm tắt các sự kiện/thông tin cốt lõi của các lượt đó thành một đoạn văn ngắn (`summary`). Đoạn tóm tắt này sau đó được lưu vào một biến trạng thái và tự động tiêm vào system message hoặc chèn ở đầu history của các lượt tiếp theo. Nhờ đó, chatbot vừa bảo toàn được ngữ cảnh dài hạn vừa không làm bùng nổ số lượng input token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
