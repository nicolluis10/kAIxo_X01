let sessionId = localStorage.getItem('sessionId') || null;

document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();
    if (message === '') return;

    // Display user's message
    addMessageToChatWindow('user', message);
    userInput.value = '';

    // Prepare payload
    const payload = {
        message: message,
        session_id: sessionId
    };

    // Send message to backend
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json()
    })
    .then(data => {
        // Update sessionId
        sessionId = data.session_id;
        localStorage.setItem('sessionId', sessionId);
        // Display bot's response
        addMessageToChatWindow('bot', data.response);
    })
    .catch(error => {
        console.error('Error:', error);
        addMessageToChatWindow('bot', 'Errore bat gertatu da. Mesedez, saiatu berriro.');
    });
}

function addMessageToChatWindow(sender, message) {
    const chatWindow = document.getElementById('chat-window');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender);

    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');
    messageContent.textContent = message;

    messageElement.appendChild(messageContent);
    chatWindow.appendChild(messageElement);

    // Scroll to the bottom
    chatWindow.scrollTop = chatWindow.scrollHeight;
}
