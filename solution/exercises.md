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
> *Việt Nam có nhiều thành phố đẹp và là một đất nước ổn định về mặt kinh tế đời sống và xã hội*
**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Khi tăng temperature từ 0.0 lên 1.5, phản hồi dịch chuyển từ mức độ hoàn toàn cố định, logic rạch ròi, lặp lại (ở 0.0) sang mức độ sáng tạo cao, đa dạng chi tiết*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Nên đặt temperature ở mức thấp (khoảng từ 0.2 đến 0.3) cho chatbot hỗ trợ khách hàng. Mức này đảm bảo câu trả lời mang tính chính xác cao, nhất quán, bám sát tài liệu hướng dẫn và hạn chế tối đa việc mô hình tự ý sáng tạo*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *Mức chênh lệch chi phí: Với khối lượng 30.000 lượt gọi/ngày và mỗi lần 350 token đầu ra, ước tính chi phí dùng model gpt-4o sẽ đắt hơn khoảng 14 đến 16 lần so với dùng gpt-4o-mini (dựa trên bảng giá đầu vào/đầu ra).Trường hợp dùng GPT-4o: Dành cho các tác vụ tư duy phức tạp, phân tích dữ liệu chuyên sâu, lập luận logic nhiều bước hoặc dịch thuật văn bản văn học đòi hỏi độ tinh tế cao.Trường hợp dùng Mini: Dành cho các tác vụ tự động hóa quy mô lớn, phân loại văn bản đơn giản, trích xuất thông tin hoặc chatbot chăm sóc khách hàng cơ bản có lưu lượng truy cập khổng lồ*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Phản hồi của hai trường hợp có sự khác biệt rõ rệt: phiên bản giáo viên tiểu học dùng từ ngữ cực kỳ gần gũi, chia nhỏ khái niệm và dùng ví dụ thực tế quen thuộc với trẻ em, trong khi phiên bản chuyên gia tài chính sử dụng thuật ngữ chuyên ngành dày đặc và cấu trúc phân tích phức tạp. Điều này cho thấy system prompt định hình quyền hạn, phong cách giao tiếp, tầng nhận thức và cách mô hình chọn lọc từ vựng để xử lý dữ liệu đầu vào*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Mức chênh lệch: Số token thực tế đếm bằng tiktoken đối với văn bản tiếng Việt thường cao hơn khoảng 30% đến 50% so với công thức ước lượng thô (số từ / 0.75).Nguyên nhân: Tiếng Việt có nhiều dấu thanh và các nguyên âm ghép. Bộ mã hóa (tokenizer) của OpenAI được huấn luyện chủ yếu trên tiếng Anh, nên các từ tiếng Việt thường bị bẻ nhỏ thành nhiều sub-word token hơn, dẫn đến việc tiêu tốn nhiều token hơn trên cùng một độ dài ký tự*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming đóng vai trò cực kỳ quan trọng trong các giao diện trò chuyện trực tiếp (chat interface, trợ lý ảo thời gian thực) nhằm loại bỏ cảm giác chờ đợi mệt mỏi cho người dùng khi mô hình đang sinh câu trả lời dài. Ngược lại, phương thức non-streaming lại phù hợp hơn trong các tác vụ tự động hóa ngầm (background jobs), gọi API qua lại giữa các hệ thống (API-to-API), hoặc khi cần xử lý, kiểm tra toàn bộ cấu trúc dữ liệu JSON trước khi hiển thị hoặc lưu trữ vào cơ sở dữ liệu* 

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Cơ chế exponential backoff giúp giảm áp lực tức thời cho hệ thống máy chủ đang quá tải bằng cách tăng dần khoảng thời gian chờ sau mỗi lần thất bại (ví dụ: $0.1s \rightarrow 0.2s \rightarrow 0.4s$). Nếu hàng nghìn client cùng cấu hình một mức delay cố định (ví dụ luôn chờ đúng 1 giây), chúng sẽ đồng loạt gửi lại request cùng một thời điểm sau mỗi giây, tạo ra hiện tượng "cơn bão retry" (thadashing/retry storm) khiến máy chủ tiếp tục sập sập sập liên tục không thể hồi phục*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Persona chọn:Trợ giảng thân thiện, chuyên hỗ trợ kỹ thuật lập trình và AI cho sinh viên.System Prompt: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt*
*giải thích lựa chọn từ ngữ:"trlời ngắn gọn": Giúp mô hình tập trung vào trọng tâm câu hỏi lập trình, tiết kiệm token phản hồi (giảm độ trễ - latency) và tránh lan man, giúp người học dễ đọc code hoặc lý thuyết giải ththí*

*"ằng tiếng Việt": Định hình ngôn ngữ giao tiếp chính xác để học viên Việt Nam dễ tiếp thu các khái niệm kỹ thuật phức tạp vốn thường bằng tiếng Anh*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất: Chỉ nhớ ngắn hạn 3 lượt chat gần nhất, mất sạch ngữ cảnh khi tắt ứng dụng*
 
 *Đề xuất thiện: Tích hợp cơ chế RAG (Retrieval-Augmented Generation) kết hợp Vector Database để tra cứu lịch sử và tài liệu dài hạh*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
