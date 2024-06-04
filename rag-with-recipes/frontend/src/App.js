import React, { useState } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const formatMessage = (text) => {
    const codeBlockRegex = /```([a-z]+)?\n([\s\S]*?)```/g;

    const parts = text.split(codeBlockRegex);

    return (
      <span>
        {parts.map((part, index) => {
          if (index % 3 === 1) { // Check if it's a language annotation (0-based index)
            // In this case, we don't render the language annotation itself
            return null; 
          } else if (index % 3 === 2) { // Check if it's a code block
            return (
              <pre key={index}>
                <code className={`language-${parts[index - 1]?.trim() || ''}`}>{part.trim()}</code>
              </pre>
            );
          } else { // Regular text
            return <span key={index}>{part}</span>;
          }
        })}
      </span>
    );
  };


  const sendMessage = async () => {
    if (input.trim() === '') return;

    const userMessage = { text: input, sender: 'user' };
    setMessages([...messages, userMessage]);

    const response = await fetch('http://localhost:8000/sendMessage', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message: input })
    });

    const data = await response.json();
    const botMessage = { text: data.reply, sender: 'bot' };
    setMessages([...messages, userMessage, botMessage]);
    setInput('');
  };

  return (
    <div id="chat-container">
      <h1>RAG with Recipes</h1>
      <div id="messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            {formatMessage(msg.text)}
          </div>
        ))}
      </div>
      <div id="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default App;
