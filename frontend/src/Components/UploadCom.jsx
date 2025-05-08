"use client";

import { useState, useRef } from "react";
import { motion } from "framer-motion";
import {
  Upload,
  FileText,
  X,
  AlertCircle,
  CheckCircle,
  Loader2,
} from "lucide-react";

const UploadCom = ({
  onFileUpload,
  acceptedFileTypes = ".csv,.xls,.xlsx",
  maxSize = 10,
}) => {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  // Convert maxSize to bytes
  const maxSizeBytes = maxSize * 1024 * 1024;

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (validateFile(droppedFile)) {
        setFile(droppedFile);
        setError("");
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (validateFile(selectedFile)) {
        setFile(selectedFile);
        setError("");
      }
    }
  };

  const validateFile = (file) => {
    // Check file type based on acceptedFileTypes
    const fileExtension = file.name.split(".").pop().toLowerCase();
    const acceptedExtensions = acceptedFileTypes
      .split(",")
      .map((type) => type.trim().replace(".", ""));

    if (!acceptedExtensions.includes(fileExtension)) {
      setError(
        `Please upload a file with one of these formats: ${acceptedFileTypes}`
      );
      return false;
    }

    // Check file size
    if (file.size > maxSizeBytes) {
      setError(`File size should not exceed ${maxSize}MB`);
      return false;
    }

    return true;
  };

  const handleRemoveFile = () => {
    setFile(null);
    setSuccess("");
    setError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file to upload");
      return;
    }

    setUploading(true);
    setError("");

    try {
      await onFileUpload(file);
      setSuccess("File uploaded successfully!");
    } catch (err) {
      console.error("Upload error:", err);
      setError(err.message || "Failed to upload file. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="w-full">
      {/* Error Message */}
      {error && (
        <motion.div
          className="mb-4 p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 flex items-start"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <AlertCircle className="h-5 w-5 text-red-500 mr-2 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        </motion.div>
      )}

      {/* Success Message */}
      {success && (
        <motion.div
          className="mb-4 p-3 rounded-lg bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 flex items-start"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-green-700 dark:text-green-300">
            {success}
          </p>
        </motion.div>
      )}

      {/* Drag & Drop Area */}
      {!file ? (
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragging
              ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20"
              : "border-gray-300 dark:border-gray-600 hover:border-indigo-400 hover:bg-gray-50 dark:hover:bg-gray-700/50"
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept={acceptedFileTypes}
            className="hidden"
          />
          <div className="flex flex-col items-center justify-center">
            <div className="mb-3 p-3 bg-indigo-100 dark:bg-indigo-900/30 rounded-full">
              <Upload className="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
            </div>
            <h3 className="text-base font-medium text-gray-900 dark:text-white mb-1">
              {isDragging
                ? "Drop your file here"
                : "Drag & Drop your file here"}
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">
              or click to browse from your computer
            </p>
            <p className="text-xs text-gray-400 dark:text-gray-500">
              Supported formats: {acceptedFileTypes.replace(/\./g, "")}
            </p>
          </div>
        </div>
      ) : (
        <div className="border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="p-2 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg mr-3">
                <FileText className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                  {file.name}
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            <button
              onClick={handleRemoveFile}
              className="p-1.5 rounded-full bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              aria-label="Remove file"
            >
              <X className="h-4 w-4 text-gray-500 dark:text-gray-400" />
            </button>
          </div>
        </div>
      )}

      {/* Upload Button */}
      <div className="mt-4">
        <motion.button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="w-full inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {uploading ? (
            <>
              <Loader2 className="animate-spin -ml-1 mr-2 h-4 w-4" />
              Uploading...
            </>
          ) : (
            <>Upload</>
          )}
        </motion.button>
      </div>
    </div>
  );
};

export default UploadCom;
