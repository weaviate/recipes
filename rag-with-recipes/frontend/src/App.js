import React, { useState, useEffect } from 'react';
import './App.css';
import weaviateLogo from './weaviate-logo-W.png';
import githubLogo from './github.png';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [requestTime, setRequestTime] = useState(0);
  const [sessionId, setSessionId] = useState(null);

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

  useEffect(() => {
    let interval = null;

    if (loading) {
      interval = setInterval(() => {
        setRequestTime((prevTime) => prevTime + 1);
      }, 1000);
    } else {
      setRequestTime(0);
    }

    return () => {
      clearInterval(interval);
    };
  }, [loading]);

  useEffect(() => {
    // Create a session when the component mounts
    const createSession = async () => {
      try {
        const response = await fetch('http://localhost:8000/create-session', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          }
        });

        const data = await response.json();
        setSessionId(data.session_id);
      } catch (error) {
        console.error('Error creating session:', error);
      }
    };

    createSession();
  }, []);

  const sendMessage = async () => {
    if (input.trim() === '') return;

    const userMessage = { text: input, sender: 'user' };
    setMessages([...messages, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/sendMessage', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: input, sessionId: sessionId })
      });

      const data = await response.json();
      const botMessage = { text: data.reply, sender: 'bot' };
      setMessages([...messages, userMessage, botMessage]);
    } catch (error) {
      console.error('Error:', error);
    }

    setLoading(false);
  };

  return (
    <div id="chat-container">
      <div id="header">
        <img src={weaviateLogo} alt="Left Image" id="left-image" />
        <h1>RAG with Recipes</h1>
        <img src={githubLogo} alt="Right Image" id="right-image" />
      </div>
      <div id="messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            {formatMessage(msg.text)}
          </div>
        ))}
        {loading && (
          <div className="loading-animation">
            <div className="spinner"></div>
            <p>Loading... {requestTime}s</p>
          </div>
        )}
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