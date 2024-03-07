import React, { useState } from 'react';
import backgroundImage from './assets/bg-light-3.png';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [topText, setTopText] = useState('');

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const handleTopTextChange = (event) => {
    setTopText(event.target.value);
  };

  const handleSendMessage = () => {
    if (inputText.trim() !== '') {
      const newUserMessage = {
        text: inputText,
        sender: 'user',
      };
      setMessages([...messages, newUserMessage]);
      setInputText('');

      // Send the chat history and top text to the backend
      const data = {
        messages: [...messages, newUserMessage],
        topText: topText,
      };
      fetch('http://localhost:8000/RAGwithPersona', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      })
        .then((response) => response.json())
        .then((data) => {
          // Add the response from the backend to the messages array
          const newBotMessage = {
            text: data.response,
            sender: 'bot',
          };
          setMessages((prevMessages) => [...prevMessages, newBotMessage]);
        })
        .catch((error) => {
          console.error('Error:', error);
        });
    }
  };

  return (
    <div className="chatbot" style={{ backgroundImage: `url(${backgroundImage})` }}>
      <h1>RAGwithPersona</h1>
      <div className="top-text">
        <input
          type="text"
          value={topText}
          onChange={handleTopTextChange}
          placeholder="Enter Persona here"
        />
      </div>
      <div className="chat-history">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
            {message.text}
          </div>
        ))}
      </div>
      <div className="input-container">
        <input
          type="text"
          value={inputText}
          onChange={handleInputChange}
          placeholder="Type your message"
        />
        <button onClick={handleSendMessage}>Send</button>
      </div>
    </div>
  );
};

export default App;