import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FiSend,
  FiX,
  FiRefreshCw,
  FiAlertCircle,
  FiMinimize2,
  FiMaximize2,
} from "react-icons/fi";
import { useParams } from "react-router-dom";
import axiosInstance from "../utils/axiosInstance";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { FaRobot } from "react-icons/fa";

const ChatBot = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState(null);
  const [isMinimized, setIsMinimized] = useState(false);
  const messagesEndRef = useRef(null);
  const { project_id } = useParams();

  // Reset state when component mounts or key changes
  useEffect(() => {
    setMessages([]);
    setInputMessage("");
    setIsLoading(false);
    setIsTyping(false);
    setError(null);
    setIsMinimized(false);
  }, [project_id]); // Reset when project_id changes

  // Add welcome message when chat is first opened
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([
        {
          type: "bot",
          content: `👋 Welcome to your AI Assistant! I'm here to help you analyze and understand your data. You can ask me questions like:

• "What are the key insights from this dataset?"
• "Show me a summary of the data"
• "What are the correlations between variables?"
• "Generate a visualization of [specific data]"
• "Explain the trends in [specific column]"

Feel free to ask any questions about your data, and I'll help you make sense of it!`,
          timestamp: new Date().toISOString(),
          status: "sent",
        },
      ]);
    }
  }, [isOpen]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen) {
      fetchChatHistory();
      // Load messages from localStorage
      const savedMessages = localStorage.getItem(`chat_messages_${project_id}`);
      if (savedMessages) {
        setMessages(JSON.parse(savedMessages));
      }
    }
  }, [project_id, isOpen]);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem(
        `chat_messages_${project_id}`,
        JSON.stringify(messages)
      );
    }
  }, [messages, project_id]);

  const fetchChatHistory = async () => {
    try {
      const response = await axiosInstance.get(
        `/projects/${project_id}/chat_history`
      );
      if (response.data && response.data.chat_history) {
        setMessages(response.data.chat_history);
      }
    } catch (error) {
      console.error("Error fetching chat history:", error);
      setError(
        error.response?.data?.detail ||
          error.message ||
          "Failed to load chat history. Please try again."
      );
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const newMessage = {
      type: "user",
      content: inputMessage,
      timestamp: new Date().toISOString(),
      status: "sent",
    };

    setMessages((prev) => [...prev, newMessage]);
    setInputMessage("");
    setIsLoading(true);
    setIsTyping(true);
    setError(null);

    try {
      const response = await axiosInstance.post(
        `/projects/${project_id}/chat`,
        {
          query: inputMessage,
        }
      );

      // Check if response has the expected structure
      if (!response.data) {
        throw new Error("No response data received");
      }

      const botMessage = {
        type: "bot",
        content:
          typeof response.data === "string"
            ? response.data
            : response.data.response || JSON.stringify(response.data),
        timestamp: new Date().toISOString(),
        status: "sent",
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      let errorMessage = "Sorry, I encountered an error. Please try again.";

      if (error.response?.status === 404) {
        errorMessage =
          "The chat endpoint is not available. Please check your connection.";
      } else if (error.response?.data?.detail?.includes("Invalid API key")) {
        errorMessage =
          "There's an issue with the API configuration. Please contact support.";
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.message) {
        errorMessage = error.message;
      }

      const errorMessageObj = {
        type: "error",
        content: errorMessage,
        timestamp: new Date().toISOString(),
        status: "error",
        originalMessage: newMessage,
      };
      setMessages((prev) => [...prev, errorMessageObj]);
      setError(errorMessage);
    } finally {
      setIsLoading(false);
      setIsTyping(false);
    }
  };

  const handleRetry = async (errorMessage) => {
    if (!errorMessage.originalMessage) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await axiosInstance.post(
        `/projects/${project_id}/chat`,
        {
          query: errorMessage.originalMessage.content,
        }
      );

      if (!response.data) {
        throw new Error("No response data received");
      }

      const botMessage = {
        type: "bot",
        content:
          typeof response.data === "string"
            ? response.data
            : response.data.response || JSON.stringify(response.data),
        timestamp: new Date().toISOString(),
        status: "sent",
      };

      // Remove the error message and add the new response
      setMessages((prev) =>
        prev.filter((msg) => msg !== errorMessage).concat(botMessage)
      );
    } catch (error) {
      console.error("Error retrying message:", error);
      setError(
        error.response?.data?.detail ||
          error.message ||
          "Failed to retry message. Please try again."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const renderMessage = (message) => {
    const isCodeBlock = (content) => {
      return content.includes("```") || content.includes("`");
    };

    if (message.type === "error") {
      return (
        <div className="flex items-center space-x-2">
          <FiAlertCircle className="text-red-500" />
          <p className="text-red-500">{message.content}</p>
          <button
            onClick={() => handleRetry(message)}
            className="text-blue-500 hover:text-blue-700"
            title="Retry message"
          >
            <FiRefreshCw />
          </button>
        </div>
      );
    }

    if (message.type === "bot") {
      return (
        <div className="flex items-start space-x-3">
          <div className="flex-shrink-0 mt-1">
            <FaRobot className="text-blue-600 text-xl" />
          </div>
          <div className="flex-1">
            {isCodeBlock(message.content) ? (
              <ReactMarkdown
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "");
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, "")}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            ) : (
              <p className="text-sm whitespace-pre-wrap">{message.content}</p>
            )}
          </div>
        </div>
      );
    }

    return <p className="text-sm">{message.content}</p>;
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-50 flex items-center justify-center"
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className={`bg-white rounded-lg shadow-xl flex flex-col ${
            isMinimized ? "w-96 h-16" : "w-4/5 h-4/5"
          } transition-all duration-300`}
        >
          {/* Chat Header */}
          <div className="p-4 bg-blue-600 text-white rounded-t-lg flex justify-between items-center">
            <h3 className="font-semibold text-lg flex items-center space-x-2">
              <FaRobot className="text-xl" />
              <span>AI Assistant</span>
            </h3>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setIsMinimized(!isMinimized)}
                className="hover:bg-blue-700 p-1 rounded-full transition-colors"
              >
                {isMinimized ? (
                  <FiMaximize2 size={20} />
                ) : (
                  <FiMinimize2 size={20} />
                )}
              </button>
              <button
                onClick={onClose}
                className="hover:bg-blue-700 p-1 rounded-full transition-colors"
              >
                <FiX size={20} />
              </button>
            </div>
          </div>

          {!isMinimized && (
            <>
              {/* Error Message */}
              {error && (
                <div className="p-2 bg-red-50 text-red-600 text-sm flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <FiAlertCircle />
                    <span>{error}</span>
                  </div>
                  <button
                    onClick={() => setError(null)}
                    className="text-red-600 hover:text-red-800"
                  >
                    <FiX size={16} />
                  </button>
                </div>
              )}

              {/* Messages Container */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((message, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex ${
                      message.type === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg p-3 ${
                        message.type === "user"
                          ? "bg-blue-600 text-white"
                          : message.type === "error"
                          ? "bg-red-50"
                          : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      {renderMessage(message)}
                      <span className="text-xs opacity-70 mt-1 block">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </motion.div>
                ))}
                {isTyping && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex justify-start"
                  >
                    <div className="bg-gray-100 rounded-lg p-3">
                      <div className="flex space-x-2">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
                      </div>
                    </div>
                  </motion.div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Form */}
              <form
                onSubmit={handleSendMessage}
                className="p-4 border-t bg-gray-50"
              >
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    placeholder="Type your message..."
                    className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    disabled={isLoading}
                  />
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    type="submit"
                    disabled={isLoading || !inputMessage.trim()}
                    className="bg-blue-600 text-white p-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
                  >
                    <FiSend size={20} />
                  </motion.button>
                </div>
              </form>
            </>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default ChatBot;
