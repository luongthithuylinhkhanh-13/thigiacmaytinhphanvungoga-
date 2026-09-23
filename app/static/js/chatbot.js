document.addEventListener('DOMContentLoaded', () => {
    const chatFab = document.getElementById('chatFab');
    const chatWindow = document.getElementById('chatWindow');
    const chatCloseBtn = document.getElementById('chatCloseBtn');
    const chatClearBtn = document.getElementById('chatClearBtn');
    const chatBody = document.getElementById('chatBody');
    const chatInput = document.getElementById('chatInput');
    const chatSendBtn = document.getElementById('chatSendBtn');
    const chatStatus = document.getElementById('chatStatus');

    // Mở / Đóng Chatbot
    chatFab.addEventListener('click', () => {
        chatWindow.classList.toggle('open');
        if (chatWindow.classList.contains('open')) {
            chatInput.focus();
        }
    });

    chatCloseBtn.addEventListener('click', () => {
        chatWindow.classList.remove('open');
    });

    // Xóa lịch sử chat (Trừ lời chào)
    chatClearBtn.addEventListener('click', () => {
        const welcomeMessage = `
            <div class="chat-message bot">
                Xin chào! Tôi là Trợ lý AI. Tôi có thể giúp bạn tìm hiểu về hệ thống phân vùng ổ gà, kết quả dự đoán và cách sử dụng website.
            </div>
        `;
        chatBody.innerHTML = welcomeMessage;
    });

    // Thêm tin nhắn vào khung chat
    function appendMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('chat-message', sender);
        msgDiv.textContent = text;
        chatBody.appendChild(msgDiv);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    // Gửi tin nhắn
    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        appendMessage(text, 'user');
        chatInput.value = '';
        chatInput.focus();

        chatStatus.textContent = "Trợ lý AI đang xử lý...";

        // Lấy Context thực tế từ giao diện
        let currentContext = "Chưa có ảnh nào được phân tích.";
        const resCount = document.getElementById('resCount');
        if (resCount && resCount.textContent !== '--') {
            const resArea = document.getElementById('resArea').textContent;
            const resConf = document.getElementById('resConf').textContent;
            const riskList = document.getElementById('riskList');
            const riskText = riskList ? riskList.innerText.replace(/\n/g, ' - ') : 'Chưa rõ';
            
            currentContext = `Số ổ gà: ${resCount.textContent}\nTổng diện tích: ${resArea}\nConfidence trung bình: ${resConf}\nĐánh giá Risk: ${riskText}`;
        }

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    message: text,
                    context: currentContext
                })
            });

            const data = await response.json().catch(() => ({}));

            if (!response.ok) {
                const errMsg = data.answer || data.reply || data.error || "Lỗi kết nối từ máy chủ.";
                appendMessage(errMsg, 'bot');
                return;
            }

            const botAnswer = data.answer || data.reply;
            if (botAnswer) {
                appendMessage(botAnswer, 'bot');
            } else {
                appendMessage("Xin lỗi, có lỗi khi nhận phản hồi từ hệ thống.", 'bot');
            }
        } catch (error) {
            console.error("Chat Error:", error);
            appendMessage("Lỗi kết nối mạng hoặc máy chủ không phản hồi.", 'bot');
        } finally {
            chatStatus.textContent = "Trợ lý AI đang hoạt động";
        }
    }

    chatSendBtn.addEventListener('click', (e) => {
        e.preventDefault();
        sendMessage();
    });

    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });
});
