import React, { useState, useRef } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { FiUpload, FiX, FiCheck, FiAlertCircle } from "react-icons/fi";
import { toast } from "react-toastify";
import axios from "axios";
import ChatBot from "../components/ChatBot";

const Upload = () => {
  // ... existing code ...

  return (
    <div className="min-h-screen bg-gray-50">
      {/* ... existing code ... */}

      {/* Add ChatBot component */}
      <ChatBot />
    </div>
  );
};

export default Upload;
