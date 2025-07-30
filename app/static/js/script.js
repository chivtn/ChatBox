function searchMessages() {
    const query = document.getElementById("message").value.trim();
    const messagesDiv = document.getElementById("messages");

    if (query === "") {
        messagesDiv.innerHTML += "<div class='alert alert-danger'>Vui lòng nhập nội dung!</div>";
        return;
    }

    // Hiển thị tin nhắn người dùng
    const userMessage = document.createElement("div");
    userMessage.className = "alert alert-secondary";
    userMessage.innerText = "Bạn: " + query;
    messagesDiv.appendChild(userMessage);

    fetch('/api/search',{
        method : 'post',
        headers : {
            'Content-Type' : 'application/json'
        },
        body : JSON.stringify({query : query})
    })
    .then(res=> res.json())
    .then(data => {
        // Hiển thị câu trả lời từ Gemini
        const geminiAnswer = document.createElement("div");
        geminiAnswer.className = "alert alert-success";
        geminiAnswer.innerHTML = `<strong>Trả lời:</strong> ${data.answer}`;
        messagesDiv.appendChild(geminiAnswer);

        // Hiển thị kết quả tham khảo
        if (data.results && data.results.length > 0) {
            const refHeader = document.createElement("div");
            refHeader.className = "alert alert-info";
            refHeader.innerHTML = "<strong>Thông tin tham khảo:</strong>";
            messagesDiv.appendChild(refHeader);

            data.results.forEach(result => {
                const botReply = document.createElement("div");
                botReply.className = "alert alert-primary";
                botReply.innerHTML = `<strong>${result.title}</strong> <br>
                                      <em>Độ tương đồng: ${result.score}</em><br>
                                      <a href="${result.href}" target="_blank">${result.href}</a>`;
                messagesDiv.appendChild(botReply);
            });
        }
    })
    .catch(error => {
        console.error('Error:', error);
        const errorMsg = document.createElement("div");
        errorMsg.className = "alert alert-danger";
        errorMsg.innerText = "Có lỗi xảy ra khi tìm kiếm";
        messagesDiv.appendChild(errorMsg);
    });

    document.getElementById("message").value = "";

    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}