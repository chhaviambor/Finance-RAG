document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    
    // Auto-scroll helper
    const scrollToBottom = () => {
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    const addMessage = (content, sender, isMarkdown = false) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        
        if (isMarkdown && sender === 'assistant') {
            bubble.innerHTML = marked.parse(content);
        } else {
            bubble.textContent = content;
        }

        msgDiv.appendChild(bubble);
        chatBox.appendChild(msgDiv);
        scrollToBottom();
    };

    const addTypingIndicator = () => {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant typing-msg';
        
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            indicator.appendChild(dot);
        }
        
        msgDiv.appendChild(indicator);
        chatBox.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    };

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const text = userInput.value.trim();
        if (!text) return;
        
        userInput.value = '';
        addMessage(text, 'user');
        
        const indicator = addTypingIndicator();
        
        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: text })
            });
            
            indicator.remove();
            
            if (!response.ok) {
                const errData = await response.json();
                addMessage(`**Error**: ${errData.detail || 'The server encountered an issue.'}`, 'assistant', true);
                return;
            }
            
            const data = await response.json();
            addMessage(data.answer, 'assistant', true);
            
        } catch (error) {
            indicator.remove();
            addMessage(`**Error**: Could not connect to the API.`, 'assistant', true);
        }
    });
});
